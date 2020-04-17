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
from modules.token_predictor import SchemaTokenPredictor


class DualSQL(nn.Module):
    """Interaction model, where an interaction is processed all at once."""

    def __init__(
            self,
            params,
            utter_vocab,
            query_vocab,
            query_vocab_schema):

        # Create embedders
        if params.use_bert: # TODO
            self.model_bert, self.tokenizer, self.bert_config = utils_bert.get_bert(params)

            utter_vocab_emb, query_vocab_emb, query_schema_vocab_emb, utter_emb_size = load_all_embs(
                utter_vocab, query_vocab, query_vocab_schema, params.embedding_filename)

            self.query_embedder = Embedder(
                params.query_emb_size,
                initializer=query_vocab_emb,
                vocabulary=query_vocab,
                freeze=False)
            self.column_name_token_embedder = None
        else:
            utter_vocab_emb, query_vocab_emb, query_schema_vocab_emb, utter_emb_size = load_all_embs(
                utter_vocab, query_vocab, query_vocab_schema, params.embedding_filename)

            params.utter_emb_size = utter_emb_size

            self.utter_embedder = Embedder(
                params.utter_emb_size,
                initializer=utter_vocab_emb,
                vocabulary=utter_vocab,
                freeze=params.freeze)

            self.query_embedder = Embedder(
                params.query_emb_size,
                initializer=query_vocab_emb,
                vocabulary=query_vocab,
                freeze=False)

            self.column_name_token_embedder = Embedder(
                params.utter_emb_size,
                initializer=query_schema_vocab_emb,
                vocabulary=query_vocab_schema,
                freeze=params.freeze)
        
        # Create the encoder
        encoder_utter_size = self.bert_config.hidden_size if params.use_bert else params.utter_emb_size
        encoder_utter_size += params.encoder_state_size / 2 # discourse-level lstm
        encoder_query_size = params.encoder_state_size
        self.utter_encoder = nn.LSTM(
            encoder_utter_size,
            encoder_query_size,
            params.encoder_num_layers)

        # Positional embedder for texts
        attention_key_size = params.encoder_state_size
        self.schema_attention_key_size = attention_key_size
        if params.state_positional_embeddings:
            attention_key_size += params.positional_embedding_size
            self.positional_embedder = Embedder(
                params.positional_embedding_size,
                num_tokens=params.maximum_utterances)

        self.utter_attention_key_size = attention_key_size

        # Create the discourse-level LSTM parameters
        self.discourse_lstms = nn.LSTM(
            params.encoder_state_size,
            params.encoder_state_size/2)
        # self.initial_discourse_state = torch_utils.add_params(tuple([params.encoder_state_size / 2]), "V-turn-state-0")

        # Previous query Encoder
        self.query_encoder = nn.LSTM(
            params.encoder_state_size,
            params.encoder_num_layers,
            params.query_emb_size)

        self.text_schema_encoder = TextSchemaEncoder(params)
        self.interaction_encoder = InteractionEncoder(params)

        token_predictor = SchemaTokenPredictor(
            params,
            query_vocab,
            self.utter_attention_key_size,
            self.schema_attention_key_size)

        # Use schema_attention in decoder
        decoder_input_size = params.query_emb_size + \
            self.text_attention_key_size + \
            self.schema_attention_key_size + \
            params.encoder_state_size
        self.decoder = SequencePredictorWithSchema(
            params,
            decoder_input_size,
            self.query_embedder,
            self.column_name_token_embedder,
            token_predictor)
        self.dropout = 0.

        self.params = params

    def forward(
            self,
            interaction,
            schema_states,
            max_gen_length,
            utter_sequence=None):

        utter_hidden_states = []

        if self.params.use_bert:
            final_utterance_state, utterance_states, schema_states = self.get_bert_encoding(
                utter_sequence, utter_schema, discourse_state, dropout=True)
        else:
            if self.params.discourse_level_lstm:
                utterance_token_embedder = lambda token: torch.cat([
                    self.utter_embedder(token), discourse_state], dim=0)
            else:
                utterance_token_embedder = self.utter_embedder
            final_utterance_state, utterance_states = self.utter_encoder(
                utter_sequence,
                utterance_token_embedder,
                dropout_amount=self.dropout)

        utter_hidden_states.extend(utterance_states)
        utter_sequences.append(utter_sequence)

        num_utterances_to_keep = min(self.params.maximum_utterances, len(utter_sequences))

        # final_utterance_state[1][0] is the first layer's hidden states at the last time step (concat forward lstm and backward lstm)
        if self.params.discourse_level_lstm:
            _, discourse_state, discourse_lstm_states = torch_utils.forward_one_multilayer(self.discourse_lstms, final_utterance_state[1][0], discourse_lstm_states, self.dropout)

        if self.params.use_utterance_attention:
            final_utterance_states_c, final_utterance_states_h, final_utterance_state = self.get_utterance_attention(final_utterance_states_c, final_utterance_states_h, final_utterance_state, num_utterances_to_keep)

        if self.params.state_positional_embeddings:
            utterance_states, flat_sequence = self._add_positional_embeddings(utter_hidden_states, utter_sequences)
        else:
            flat_sequence = []
            for utt in utter_sequences[-num_utterances_to_keep:]:
                flat_sequence.extend(utt)

        if self.params.use_previous_query and len(prev_query) > 0:
            previous_queries, previous_query_states = self.get_previous_queries(previous_queries, previous_query_states, prev_query, utter_schema)

        if len(gold_query) <= max_gen_length and len(prev_query) <= max_gen_length:
            predicted_sequence = []
            fed_sequence = []

            decoder_results = self.decoder(
                text_final_state,
                utter_hidden_states,
                schema_states,
                max_gen_length,
                utter_sequence=utter_sequence,
                previous_queries=previous_queries,
                previous_query_states=previous_query_states,
                utter_schema=utter_schema,
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
            utter_hidden_states,
            schema_states,
            max_gen_length,
            gold_sequence=gold_query,
            utter_sequence=utter_sequence,
            previous_queries=previous_queries,
            previous_query_states=previous_query_states,
            utter_schema=utter_schema,
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

    def get_previous_queries(self, previous_queries, previous_query_states, prev_query, utter_schema):
        """
        """
        previous_queries.append(prev_query)
        num_queries_to_keep = min(self.params.maximum_queries, len(previous_queries))
        previous_queries = previous_queries[-num_queries_to_keep:]

        query_token_embedder = lambda query_token: self.get_query_token_embedding(query_token, utter_schema)
        _, previous_querys = self.query_encoder(prev_query, query_token_embedder, dropout_amount=self.dropout)
        assert len(previous_querys) == len(prev_query)
        previous_query_states.append(previous_querys)
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

        utter_hidden_states = []
        utter_sequences = []

        final_text_states_c = []
        final_text_states_h = []

        previous_query_states = []
        previous_queries = []

        decoder_states = []

        discourse_state = None
        discourse_state, discourse_lstm_states = self._initialize_discourse_states()
        discourse_states = []

        # Schema and schema embeddings
        utter_schema = interaction.get_schema()
        schema_states = []

        if utter_schema:
            schema_states = self.encode_schema_bow_simple(utter_schema)

        if self.utterance2query:
            gold_texts = interaction.gold_utterances()
        else:
            gold_texts = interaction.gold_queries()
            
        for text in gold_texts:
            utter_sequence = text.utter_sequence()
            prev_query = text.prev_query()
            
            # Get the gold query: reconstruct if the alignment probability is less than one
            gold_query = text.gold_query()

            # Encode the text, and update the discourse-level states
            if not self.params.use_bert:
                if self.params.discourse_level_lstm:
                    text_token_embedder = lambda token: torch.cat([self.utter_embedder(token), discourse_state], dim=0)
                else:
                    text_token_embedder = self.utter_embedder
                final_text_state, text_states = self.text_encoder(
                    utter_sequence,
                    text_token_embedder,
                    dropout_amount=self.dropout)
            else:
                final_text_state, text_states, schema_states = self.get_bert_encoding(utter_sequence, utter_schema, discourse_state, dropout=True)

            utter_hidden_states.extend(text_states)
            utter_sequences.append(utter_sequence)

            num_texts_to_keep = min(self.params.maximum_texts, len(utter_sequences))

            # final_text_state[1][0] is the first layer's hidden states at the last time step (concat forward lstm and backward lstm)
            _, discourse_state, discourse_lstm_states = torch_utils.forward_one_multilayer(self.discourse_lstms, final_text_state[1][0], discourse_lstm_states, self.dropout)

            if self.params.use_text_attention:
                final_text_states_c, final_text_states_h, final_text_state = self.get_text_attention(final_text_states_c, final_text_states_h, final_text_state, num_texts_to_keep)

            # state_positional_embeddings
            text_states, flat_sequence = self._add_positional_embeddings(utter_hidden_states, utter_sequences)

            if len(prev_query) > 0:
                previous_queries, previous_query_states = self.get_previous_queries(previous_queries, previous_query_states, prev_query, utter_schema)

            if len(gold_query) <= max_gen_length and len(prev_query) <= max_gen_length:
                loss, decoder_states = self.loss_turn(
                    final_text_state,
                    text_states,
                    schema_states,
                    max_gen_length,
                    gold_query=gold_query,
                    utter_sequence=flat_sequence,
                    previous_queries=previous_queries,
                    previous_query_states=previous_query_states,
                    utter_schema=utter_schema,
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

        utter_hidden_states = []
        utter_sequences = []

        final_text_states_c = []
        final_text_states_h = []

        previous_query_states = []
        previous_queries = []

        discourse_state = None
        if self.params.discourse_level_lstm:
            discourse_state, discourse_lstm_states = self._initialize_discourse_states()
        discourse_states = []

        # Schema and schema embeddings
        utter_schema = interaction.get_schema()
        schema_states = []

        if utter_schema and not self.params.use_bert:
            schema_states = self.encode_schema_bow_simple(utter_schema)

        interaction.start_interaction()
        while not interaction.done():
            text = interaction.next_text()

            prev_query = text.prev_query()

            utter_sequence = text.utter_sequence()

            if not self.params.use_bert:
                if self.params.discourse_level_lstm:
                    text_token_embedder = lambda token: torch.cat([self.utter_embedder(token), discourse_state], dim=0)
                else:
                    text_token_embedder = self.utter_embedder
                final_text_state, text_states = self.text_encoder(
                    utter_sequence,
                    text_token_embedder)
            else:
                final_text_state, text_states, schema_states = self.get_bert_encoding(utter_sequence, utter_schema, discourse_state, dropout=False)

            utter_hidden_states.extend(text_states)
            utter_sequences.append(utter_sequence)

            num_texts_to_keep = min(self.params.maximum_texts, len(utter_sequences))

            if self.params.discourse_level_lstm:
                _, discourse_state, discourse_lstm_states = torch_utils.forward_one_multilayer(self.discourse_lstms, final_text_state[1][0], discourse_lstm_states)

            if self.params.use_text_attention:
               final_text_states_c, final_text_states_h, final_text_state = self.get_text_attention(final_text_states_c, final_text_states_h, final_text_state, num_texts_to_keep)

            if self.params.state_positional_embeddings:
                text_states, flat_sequence = self._add_positional_embeddings(utter_hidden_states, utter_sequences)
            else:
                flat_sequence = []
                for utt in utter_sequences[-num_texts_to_keep:]:
                    flat_sequence.extend(utt)

            if self.params.use_previous_query and len(prev_query) > 0:
                previous_queries, previous_query_states = self.get_previous_queries(previous_queries, previous_query_states, prev_query, utter_schema)

            results = self.predict_turn(final_text_state,
                                        text_states,
                                        schema_states,
                                        max_gen_length,
                                        utter_sequence=flat_sequence,
                                        previous_queries=previous_queries,
                                        previous_query_states=previous_query_states,
                                        utter_schema=utter_schema)

            predicted_sequence = results[0]
            predictions.append(results)

        return predictions

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        print("Loaded model from file " + filename)
