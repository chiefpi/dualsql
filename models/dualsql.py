"""Symmetric model for dual learning."""
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_utils
import utils_bert

from data_utils.vocab import EOS_TOK, UNK_TOK

from modules.embedder import load_all_embs, Embedder
from modules.text_schema_encoder import TextSchemaEncoder
from modules.interaction_encoder import InteractionEncoder
from modules.decoder import SequencePredictorWithSchema
from modules.token_predictor import construct_token_predictor


class DualSQL(nn.Module):
    """Interaction model, where an interaction is processed all at once."""

    def __init__(
            self,
            params,
            input_vocab,
            output_vocab,
            output_vocab_schema):

        if params.use_bert:
            self.model_bert, self.tokenizer, self.bert_config = utils_bert.get_bert(params)

            input_vocab_emb, output_vocab_emb, output_schema_vocab_emb, input_emb_size = load_all_embs(
                input_vocab, output_vocab, output_vocab_schema, params.embedding_filename)

            # Create the output embeddings
            self.output_embedder = Embedder(
                params.output_emb_size,
                name="output-embedding", # TODO: delete names
                initializer=output_vocab_emb,
                vocabulary=output_vocab,
                freeze=False)
            self.column_name_token_embedder = None
        else:
            input_vocab_emb, output_vocab_emb, output_schema_vocab_emb, input_emb_size = load_all_embs(
                input_vocab, output_vocab, output_vocab_schema, params.embedding_filename)

            params.input_emb_size = input_emb_size

            # Create the input embeddings
            self.input_embedder = Embedder(
                params.input_emb_size,
                name="input-embedding",
                initializer=input_vocab_emb,
                vocabulary=input_vocab,
                freeze=params.freeze)

            # Create the output embeddings
            self.output_embedder = Embedder(
                params.output_emb_size,
                name="output-embedding",
                initializer=output_vocab_emb,
                vocabulary=output_vocab,
                freeze=False)

            self.column_name_token_embedder = Embedder(
                params.input_emb_size,
                name="schema-embedding",
                initializer=output_schema_vocab_emb,
                vocabulary=output_vocab_schema,
                freeze=params.freeze)
        
        # Create the encoder
        encoder_input_size = self.bert_config.hidden_size if params.use_bert else params.input_emb_size
        encoder_output_size = params.encoder_state_size
        encoder_input_size += params.encoder_state_size / 2 # discourse-level lstm
        self.utterance_encoder = Encoder(params.encoder_num_layers, encoder_input_size, encoder_output_size)

        # Positional embedder for utterances
        attention_key_size = params.encoder_state_size
        self.schema_attention_key_size = attention_key_size
        if params.state_positional_embeddings:
            attention_key_size += params.positional_embedding_size
            self.positional_embedder = Embedder(
                params.positional_embedding_size,
                name="positional-embedding",
                num_tokens=params.maximum_utterances)

        self.utterance_attention_key_size = attention_key_size

        # Create the discourse-level LSTM parameters
        self.discourse_lstms = torch_utils.create_multilayer_lstm_params(1, params.encoder_state_size, params.encoder_state_size / 2, "LSTM-t")
        self.initial_discourse_state = torch_utils.add_params(tuple([params.encoder_state_size / 2]), "V-turn-state-0")

        # Previous query Encoder
        self.query_encoder = Encoder(
            params.encoder_num_layers,
            params.output_emb_size,
            params.encoder_state_size)

        self.text_schema_encoder = TextSchemaEncoder(params)
        self.interaction_encoder = InteractionEncoder(params)
        self.decoder = Decoder(params)

        self.token_predictor = construct_token_predictor(
            params,
            output_vocab,
            self.text_attention_key_size,
            self.schema_attention_key_size)

        # Use schema_attention in decoder
        decoder_input_size = params.output_emb_size + self.text_attention_key_size + self.schema_attention_key_size + params.encoder_state_size
        self.decoder = SequencePredictorWithSchema(params, decoder_input_size, self.output_embedder, self.column_name_token_embedder, self.token_predictor)
        self.dropout = 0.

        self.params = params

    def predict_turn(
            self,
            text_final_state,
            input_hidden_states,
            schema_states,
            max_gen_length,
            input_sequence=None,
            previous_queries=None,
            previous_query_states=None,
            input_schema=None,
            feed_gold_tokens=False,
            training=False):
        """Predicts for a single turn."""

        predicted_sequence = []
        fed_sequence = []

        decoder_results = self.decoder(
            text_final_state,
            input_hidden_states,
            schema_states,
            max_gen_length,
            input_sequence=input_sequence,
            previous_queries=previous_queries,
            previous_query_states=previous_query_states,
            input_schema=input_schema,
            dropout_amount=self.dropout)
        predicted_sequence = decoder_results.sequence
        fed_sequence = predicted_sequence

        decoder_states = [pred.decoder_state for pred in decoder_results.predictions]

        # fed_sequence contains EOS, which we don't need when encoding snippets.
        # also ignore the first state, as it contains the BEG encoding.
        for token, state in zip(fed_sequence[:-1], decoder_states[1:]):
            if snippet_handler.is_snippet(token):
                snippet_length = 0
                for snippet in snippets:
                    if snippet.name == token:
                        snippet_length = len(snippet.sequence)
                        break
                assert snippet_length > 0
                decoder_states.extend([state for _ in range(snippet_length)])
            else:
                decoder_states.append(state)

        return (
            predicted_sequence,
            decoder_states,
            decoder_results)

    def loss_turn(
            self,
            gold_query=None,
            ):
        """Calculates loss for a single turn."""
        token_accuracy = 0.
        decoder_results = self.decoder(
            text_final_state,
            input_hidden_states,
            schema_states,
            max_gen_length,
            gold_sequence=gold_query,
            input_sequence=input_sequence,
            previous_queries=previous_queries,
            previous_query_states=previous_query_states,
            input_schema=input_schema,
            dropout_amount=self.dropout)

        all_scores = []
        all_alignments = []
        for prediction in decoder_results.predictions:
            scores = F.softmax(prediction.scores, dim=0)
            alignments = prediction.aligned_tokens
            if self.params.use_previous_query and self.params.use_copy_switch and len(previous_queries) > 0:
                query_scores = F.softmax(prediction.query_scores, dim=0)
                copy_switch = prediction.copy_switch
                scores = torch.cat([scores * (1 - copy_switch), query_scores * copy_switch], dim=0)
                alignments = alignments + prediction.query_tokens

            all_scores.append(scores)
            all_alignments.append(alignments)

        # Compute the loss
        gold_sequence = gold_query

        loss = torch_utils.compute_loss(gold_sequence, all_scores, all_alignments, get_token_indices)
        if not training:
            predicted_sequence = torch_utils.get_seq_from_scores(all_scores, all_alignments)
            token_accuracy = torch_utils.per_token_accuracy(gold_sequence, predicted_sequence)
        fed_sequence = gold_sequence

        return loss, token_accuracy, decoder_states

    def _initialize_discourse_states(self):
        discourse_state = self.initial_discourse_state

        discourse_lstm_states = []
        for lstm in self.discourse_lstms:
            hidden_size = lstm.weight_hh.size()[1]
            if lstm.weight_hh.is_cuda:
                h_0 = torch.cuda.FloatTensor(1,hidden_size).fill_(0)
                c_0 = torch.cuda.FloatTensor(1,hidden_size).fill_(0)
            else:
                h_0 = torch.zeros(1,hidden_size)
                c_0 = torch.zeros(1,hidden_size)
            discourse_lstm_states.append((h_0, c_0))

        return discourse_state, discourse_lstm_states

    def _add_positional_embeddings(self, hidden_states, utterances, group=False):
        grouped_states = []

        start_index = 0
        for utterance in utterances:
            grouped_states.append(hidden_states[start_index:start_index + len(utterance)])
            start_index += len(utterance)
        assert len(hidden_states) == sum([len(seq) for seq in grouped_states]) == sum([len(utterance) for utterance in utterances])

        new_states = []
        flat_sequence = []

        num_utterances_to_keep = min(self.params.maximum_utterances, len(utterances))
        for i, (states, utterance) in enumerate(zip(
                grouped_states[-num_utterances_to_keep:], utterances[-num_utterances_to_keep:])):
            positional_sequence = []
            index = num_utterances_to_keep - i - 1

            for state in states:
                positional_sequence.append(torch.cat([state, self.positional_embedder(index)], dim=0))

            assert len(positional_sequence) == len(utterance), \
                "Expected utterance and state sequence length to be the same, " \
                + "but they were " + str(len(utterance)) \
                + " and " + str(len(positional_sequence))

            if group:
                new_states.append(positional_sequence)
            else:
                new_states.extend(positional_sequence)
            flat_sequence.extend(utterance)

        return new_states, flat_sequence

    def get_previous_queries(self, previous_queries, previous_query_states, previous_query, input_schema):
        """
        """
        previous_queries.append(previous_query)
        num_queries_to_keep = min(self.params.maximum_queries, len(previous_queries))
        previous_queries = previous_queries[-num_queries_to_keep:]

        query_token_embedder = lambda query_token: self.get_query_token_embedding(query_token, input_schema)
        _, previous_outputs = self.query_encoder(previous_query, query_token_embedder, dropout_amount=self.dropout)
        assert len(previous_outputs) == len(previous_query)
        previous_query_states.append(previous_outputs)
        previous_query_states = previous_query_states[-num_queries_to_keep:]

        return previous_queries, previous_query_states

    def loss(self, interaction, max_gen_length, snippet_align_prob=1.):
        """Calculates the loss on a single interaction.

        Args:
            interaction (InteractionItem): An interaction to train on.
            max_gen_length (int): Maximum generation length.

        Returns:
            loss (float)
        """
        losses = []
        total_gold_tokens = 0

        input_hidden_states = []
        input_sequences = []

        final_text_states_c = []
        final_text_states_h = []

        previous_query_states = []
        previous_queries = []

        decoder_states = []

        discourse_state = None
        discourse_state, discourse_lstm_states = self._initialize_discourse_states()
        discourse_states = []

        # Schema and schema embeddings
        input_schema = interaction.get_schema()
        schema_states = []

        if input_schema:
            schema_states = self.encode_schema_bow_simple(input_schema)

        if self.utterance2query:
            gold_texts = interaction.gold_utterances()
        else:
            gold_texts = interaction.gold_queries()
            
        for text in gold_texts:
            input_sequence = text.input_sequence()
            previous_query = text.previous_query()
            
            # Get the gold query: reconstruct if the alignment probability is less than one
            gold_query = text.gold_query()

            # Encode the text, and update the discourse-level states
            if not self.params.use_bert:
                if self.params.discourse_level_lstm:
                    text_token_embedder = lambda token: torch.cat([self.input_embedder(token), discourse_state], dim=0)
                else:
                    text_token_embedder = self.input_embedder
                final_text_state, text_states = self.text_encoder(
                    input_sequence,
                    text_token_embedder,
                    dropout_amount=self.dropout)
            else:
                final_text_state, text_states, schema_states = self.get_bert_encoding(input_sequence, input_schema, discourse_state, dropout=True)

            input_hidden_states.extend(text_states)
            input_sequences.append(input_sequence)

            num_texts_to_keep = min(self.params.maximum_texts, len(input_sequences))

            # final_text_state[1][0] is the first layer's hidden states at the last time step (concat forward lstm and backward lstm)
            _, discourse_state, discourse_lstm_states = torch_utils.forward_one_multilayer(self.discourse_lstms, final_text_state[1][0], discourse_lstm_states, self.dropout)

            if self.params.use_text_attention:
                final_text_states_c, final_text_states_h, final_text_state = self.get_text_attention(final_text_states_c, final_text_states_h, final_text_state, num_texts_to_keep)

            # state_positional_embeddings
            text_states, flat_sequence = self._add_positional_embeddings(input_hidden_states, input_sequences)

            if len(previous_query) > 0:
                previous_queries, previous_query_states = self.get_previous_queries(previous_queries, previous_query_states, previous_query, input_schema)

            if len(gold_query) <= max_gen_length and len(previous_query) <= max_gen_length:
                loss, decoder_states = self.loss_turn(
                    final_text_state,
                    text_states,
                    schema_states,
                    max_gen_length,
                    gold_query=gold_query,
                    input_sequence=flat_sequence,
                    previous_queries=previous_queries,
                    previous_query_states=previous_query_states,
                    input_schema=input_schema,
                    feed_gold_tokens=True,
                    training=True)
                total_gold_tokens += len(gold_query)
                losses.append(loss)
            else:
                # sequence was too long to run the decoder
                continue

            torch.cuda.empty_cache()

        if losses:
            average_loss = torch.sum(torch.stack(losses)) / total_gold_tokens

            # Renormalize so the effect is normalized by the batch size.
            normalized_loss = average_loss
            if self.params.reweight_batch:
                normalized_loss = len(losses) * average_loss / float(self.params.batch_size)

            normalized_loss.backward()
            self.trainer.step()
            if self.params.fine_tune_bert:
                self.bert_trainer.step()
            self.zero_grad()

            loss_scalar = normalized_loss.item()
        else:
            loss_scalar = 0.

        return loss_scalar

    def predict_with_predicted_queries(self, interaction, max_gen_length, syntax_restrict=True):
        """Predicts an interaction, using the predicted queries to get snippets."""
        # assert self.params.discourse_level_lstm

        syntax_restrict=False

        predictions = []

        input_hidden_states = []
        input_sequences = []

        final_text_states_c = []
        final_text_states_h = []

        previous_query_states = []
        previous_queries = []

        discourse_state = None
        if self.params.discourse_level_lstm:
            discourse_state, discourse_lstm_states = self._initialize_discourse_states()
        discourse_states = []

        # Schema and schema embeddings
        input_schema = interaction.get_schema()
        schema_states = []

        if input_schema and not self.params.use_bert:
            schema_states = self.encode_schema_bow_simple(input_schema)

        interaction.start_interaction()
        while not interaction.done():
            text = interaction.next_text()

            previous_query = text.previous_query()

            input_sequence = text.input_sequence()

            if not self.params.use_bert:
                if self.params.discourse_level_lstm:
                    text_token_embedder = lambda token: torch.cat([self.input_embedder(token), discourse_state], dim=0)
                else:
                    text_token_embedder = self.input_embedder
                final_text_state, text_states = self.text_encoder(
                    input_sequence,
                    text_token_embedder)
            else:
                final_text_state, text_states, schema_states = self.get_bert_encoding(input_sequence, input_schema, discourse_state, dropout=False)

            input_hidden_states.extend(text_states)
            input_sequences.append(input_sequence)

            num_texts_to_keep = min(self.params.maximum_texts, len(input_sequences))

            if self.params.discourse_level_lstm:
                _, discourse_state, discourse_lstm_states = torch_utils.forward_one_multilayer(self.discourse_lstms, final_text_state[1][0], discourse_lstm_states)

            if self.params.use_text_attention:
               final_text_states_c, final_text_states_h, final_text_state = self.get_text_attention(final_text_states_c, final_text_states_h, final_text_state, num_texts_to_keep)

            if self.params.state_positional_embeddings:
                text_states, flat_sequence = self._add_positional_embeddings(input_hidden_states, input_sequences)
            else:
                flat_sequence = []
                for utt in input_sequences[-num_texts_to_keep:]:
                    flat_sequence.extend(utt)

            if self.params.use_previous_query and len(previous_query) > 0:
                previous_queries, previous_query_states = self.get_previous_queries(previous_queries, previous_query_states, previous_query, input_schema)

            results = self.predict_turn(final_text_state,
                                        text_states,
                                        schema_states,
                                        max_gen_length,
                                        input_sequence=flat_sequence,
                                        previous_queries=previous_queries,
                                        previous_query_states=previous_query_states,
                                        input_schema=input_schema)

            predicted_sequence = results[0]
            predictions.append(results)

        return predictions

    def predict_with_gold_queries(self, interaction, max_gen_length, feed_gold_query=False):
        """ Predicts SQL queries for an interaction.

        Inputs:
            interaction (Interaction): Interaction to predict for.
            feed_gold_query (bool): Whether or not to feed the gold token to the
                generation step.
        """
        # assert self.params.discourse_level_lstm

        predictions = []

        input_hidden_states = []
        input_sequences = []

        final_text_states_c = []
        final_text_states_h = []

        previous_query_states = []
        previous_queries = []

        decoder_states = []

        discourse_state = None
        if self.params.discourse_level_lstm:
            discourse_state, discourse_lstm_states = self._initialize_discourse_states()
        discourse_states = []

        # Schema and schema embeddings
        input_schema = interaction.get_schema()
        schema_states = []
        if input_schema and not self.params.use_bert:
            schema_states = self.encode_schema_bow_simple(input_schema)

        for text in interaction.gold_texts():
            input_sequence = text.input_sequence()

            previous_query = text.previous_query()

            # Encode the text, and update the discourse-level states
            if not self.params.use_bert:
                if self.params.discourse_level_lstm:
                    text_token_embedder = lambda token: torch.cat([self.input_embedder(token), discourse_state], dim=0)
                else:
                    text_token_embedder = self.input_embedder
                final_text_state, text_states = self.text_encoder(
                    input_sequence,
                    text_token_embedder,
                    dropout_amount=self.dropout)
            else:
                final_text_state, text_states, schema_states = self.get_bert_encoding(input_sequence, input_schema, discourse_state, dropout=True)

            input_hidden_states.extend(text_states)
            input_sequences.append(input_sequence)

            num_texts_to_keep = min(self.params.maximum_texts, len(input_sequences))

            if self.params.discourse_level_lstm:
                _, discourse_state, discourse_lstm_states = torch_utils.forward_one_multilayer(self.discourse_lstms, final_text_state[1][0], discourse_lstm_states, self.dropout)

            if self.params.use_text_attention:
                final_text_states_c, final_text_states_h, final_text_state = self.get_text_attention(final_text_states_c, final_text_states_h, final_text_state, num_texts_to_keep)

            if self.params.state_positional_embeddings:
                text_states, flat_sequence = self._add_positional_embeddings(input_hidden_states, input_sequences)
            else:
                flat_sequence = []
                for utt in input_sequences[-num_texts_to_keep:]:
                    flat_sequence.extend(utt)

            if self.params.use_previous_query and len(previous_query) > 0:
                previous_queries, previous_query_states = self.get_previous_queries(previous_queries, previous_query_states, previous_query, input_schema)

            prediction = self.predict_turn(
                final_text_state,
                text_states,
                schema_states,
                max_gen_length,
                gold_query=text.gold_query(),
                input_sequence=flat_sequence,
                previous_queries=previous_queries,
                previous_query_states=previous_query_states,
                input_schema=input_schema,
                feed_gold_tokens=feed_gold_query)

            decoder_states = prediction[3]
            predictions.append(prediction)

        return predictions

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        print("Loaded model from file " + filename)
