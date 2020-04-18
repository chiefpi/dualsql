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
        utter_vocab_emb, query_vocab_emb, query_schema_vocab_emb, utter_emb_size = load_all_embs(
            utter_vocab, query_vocab, query_vocab_schema, params.embedding_filename)

        if params.use_bert: # TODO: bert
            self.model_bert, self.tokenizer, self.bert_config = utils_bert.get_bert(params)
            self.column_name_token_embedder = None
        else:
            params.utter_emb_size = utter_emb_size
            
            self.utter_embedder = Embedder(
                params.utter_emb_size,
                init=utter_vocab_emb,
                vocab=utter_vocab,
                freeze=params.freeze)

            self.column_name_token_embedder = Embedder(
                params.utter_emb_size,
                init=query_schema_vocab_emb,
                vocab=query_vocab_schema,
                freeze=params.freeze)

        self.query_embedder = Embedder(
            params.query_emb_size,
            init=query_vocab_emb,
            vocab=query_vocab,
            freeze=False)

        # Create encoders
        encoder_utter_size = self.bert_config.hidden_size if params.use_bert else params.utter_emb_size
        encoder_utter_size += params.encoder_state_size / 2 # discourse-level lstm
        encoder_query_size = params.encoder_state_size
        self.utter_encoder = nn.LSTM(
            encoder_utter_size,
            encoder_query_size,
            params.encoder_num_layers,
            dropout=params.dropout)

        # Positional embedder for texts
        attention_key_size = params.encoder_state_size
        self.schema_attention_key_size = attention_key_size
        if params.state_positional_embeddings:
            attention_key_size += params.positional_embedding_size
            self.positional_embedder = Embedder(
                params.positional_embedding_size,
                num_tokens=params.maximum_utters)

        self.utter_attention_key_size = attention_key_size

        # Create the discourse-level LSTM parameters
        self.discourse_lstm = nn.LSTM(
            params.encoder_state_size,
            params.encoder_state_size/2,
            dropout=params.dropout)
        # self.initial_discourse_state = torch_utils.add_params(tuple([params.encoder_state_size / 2]), "V-turn-state-0")

        # Previous query Encoder
        self.query_encoder = nn.LSTM(
            params.encoder_state_size,
            params.encoder_num_layers,
            params.query_emb_size,
            dropout=params.dropout)

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
        # self.utter_encoder

        self.dropout = params.dropout
        self.params = params # TODO: remove

    def forward(
            self,
            primal,
            interaction,
            max_gen_length):
        """Forwards on a single interaction.

        Args:
            primal (bool): Direction of forward.
            interaction (Interaction)
            max_gen_length (int)

        Returns:
            output_seqs (list of str)
        """
        schema = interaction.schema
        all_input_seqs = interaction.utter_seqs() if primal else interaction.query_seqs()

        # History
        input_seqs = []
        input_states = []
        output_seqs = []
        prev_output_states = []
        decoder_states = []

        if self.params.discourse_level_lstm:
            discourse_state, discourse_lstm_states = self._initialize_discourse_states()

        # Schema and schema embeddings
        schema_states = []
        if schema and not self.params.use_bert:
            for column_name in schema.column_names_embedder_input:
                schema_states.append(schema.column_name_embedder_bow(
                    column_name,
                    surface_form=False,
                    column_name_token_embedder=self.column_name_token_embedder))
            schema.set_column_name_embeddings(schema_states)
            return schema_states

        for i, input_seq in enumerate(all_input_seqs):
            if self.params.use_bert: # TODO: bert works for sql?
                final_input_state, new_input_states, schema_states = self.get_bert_encoding(
                    input_seq, schema, discourse_state, dropout=True)
            else:
                input_token_embedder = lambda token: torch.cat(
                    [self.input_embedder(token), discourse_state], dim=0)
                if primal:
                    final_input_state, new_input_states = self.utter_encoder(
                        input_seq, input_token_embedder)
                else:
                    final_input_state, new_input_states = self.query_encoder(
                        input_seq, input_token_embedder)

            input_seqs.append(input_seq)
            input_states.extend(new_input_states)

            num_turns_to_keep = min(self.params.max_turns_to_keep, i+1)

            # final_input_state[1][0] is the first layer's hidden states at the last time step (concat forward lstm and backward lstm)
            _, discourse_state, discourse_lstm_states = self.discourse_lstm(
                final_input_state[1][0],
                discourse_lstm_states)

            final_utter_states_c, final_utter_states_h, final_input_state = self.get_utter_attention(
                final_utter_states_c, final_utter_states_h, final_input_state, num_turns_to_keep)

            if self.params.use_state_positional_embedding:
                input_states, flat_seq = self._add_positional_embeddings(
                    input_states, input_seqs)
            else:
                flat_seq = []
                for utt in input_seqs[-num_turns_to_keep:]:
                    flat_seq.extend(utt)

            # Decoding
            if primal:
                decoder_results = self.query_decoder(
                    final_input_state,
                    input_states,
                    schema_states,
                    max_gen_length,
                    input_seq=input_seq,
                    prev_outputs=prev_outputs,
                    prev_query_states=prev_query_states,
                    schema=schema)
            else:
                decoder_results = self.utter_decoder(
                    final_input_state,
                    input_states,
                    schema_states,
                    max_gen_length,
                    input_seq=input_seq,
                    prev_outputs=prev_outputs,
                    prev_query_states=prev_query_states,
                    schema=schema)
            predicted_seq = decoder_results.seq
            fed_seq = predicted_seq
                
            decoder_states = [pred.decoder_state for pred in decoder_results.predictions]

            if self.params.use_editing and i > 0:
                prev_outputs.append(prev_output)
                num_queries_to_keep = min(self.params.maximum_queries, len(prev_outputs))
                prev_outputs = prev_outputs[-num_queries_to_keep:]
                query_token_embedder = lambda query_token: self.get_query_token_embedding(query_token, schema)
                _, prev_querys = self.query_encoder(prev_output, query_token_embedder, dropout=self.dropout)
                assert len(prev_querys) == len(prev_output)
                prev_query_states.append(prev_querys)
                prev_query_states = prev_query_states[-num_queries_to_keep:]
                
            torch.cuda.empty_cache()

        return predicted_seq, decoder_states, decoder_results

    def loss_turn(
            self,
            gold_query=None,
            ):
        """Calculates loss for a single turn."""
        token_accuracy = 0.
        decoder_results = None

        all_scores = []
        all_alignments = []
        for prediction in decoder_results.predictions:
            scores = F.softmax(prediction.scores, dim=0)
            alignments = prediction.aligned_tokens
            if self.params.use_prev_query and self.params.use_copy_switch and len(prev_outputs) > 0:
                query_scores = F.softmax(prediction.query_scores, dim=0)
                copy_switch = prediction.copy_switch
                scores = torch.cat([scores * (1 - copy_switch), query_scores * copy_switch], dim=0)
                alignments = alignments + prediction.query_tokens

            all_scores.append(scores)
            all_alignments.append(alignments)

        # Compute the loss
        gold_seq = gold_query

        loss = torch_utils.compute_loss(gold_seq, all_scores, all_alignments, get_token_indices)
        if not training:
            predicted_seq = torch_utils.get_seq_from_scores(all_scores, all_alignments)
            token_accuracy = torch_utils.per_token_accuracy(gold_seq, predicted_seq)
        fed_seq = gold_seq

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

    def _add_positional_embeddings(self, hidden_states, utters, group=False):
        grouped_states = []

        start_index = 0
        for utter in utters:
            grouped_states.append(hidden_states[start_index:start_index + len(utter)])
            start_index += len(utter)
        assert len(hidden_states) == sum([len(seq) for seq in grouped_states]) == sum([len(utter) for utter in utters])

        new_states = []
        flat_seq = []

        num_turns_to_keep = min(self.params.maximum_utters, len(utters))
        for i, (states, utter) in enumerate(zip(
                grouped_states[-num_turns_to_keep:], utters[-num_turns_to_keep:])):
            positional_seq = []
            index = num_turns_to_keep - i - 1

            for state in states:
                positional_seq.append(torch.cat([state, self.positional_embedder(index)], dim=0))

            assert len(positional_seq) == len(utter), \
                "Expected utter and state seq length to be the same, " \
                + "but they were " + str(len(utter)) \
                + " and " + str(len(positional_seq))

            if group:
                new_states.append(positional_seq)
            else:
                new_states.extend(positional_seq)
            flat_seq.extend(utter)

        return new_states, flat_seq

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
        utter_seqs = []

        final_text_states_c = []
        final_text_states_h = []

        prev_query_states = []
        prev_outputs = []

        decoder_states = []

        discourse_state = None
        discourse_state, discourse_lstm_states = self._initialize_discourse_states()
        discourse_states = []

        # Schema and schema embeddings
        schema = interaction.get_schema()
        schema_states = []

        if schema:
            schema_states = self.encode_schema_bow_simple(schema)

        if self.utter2query:
            gold_texts = interaction.gold_utters()
        else:
            gold_texts = interaction.gold_queries()
            
        for text in gold_texts:
            utter_seq = text.utter_seq()
            prev_output = text.prev_output()
            
            # Get the gold query: reconstruct if the alignment probability is less than one
            gold_query = text.gold_query()

            # Encode the text, and update the discourse-level states
            if not self.params.use_bert:
                if self.params.discourse_level_lstm:
                    text_token_embedder = lambda token: torch.cat(
                        [self.utter_embedder(token), discourse_state], dim=0)
                else:
                    text_token_embedder = self.utter_embedder
                final_text_state, text_states = self.text_encoder(
                    utter_seq, text_token_embedder, dropout=self.dropout)
            else:
                final_text_state, text_states, schema_states = self.get_bert_encoding(utter_seq, schema, discourse_state, dropout=True)

            utter_hidden_states.extend(text_states)
            utter_seqs.append(utter_seq)

            num_texts_to_keep = min(self.params.maximum_texts, len(utter_seqs))

            # final_text_state[1][0] is the first layer's hidden states at the last time step (concat forward lstm and backward lstm)
            _, discourse_state, discourse_lstm_states = torch_utils.forward_one_multilayer(self.discourse_lstms, final_text_state[1][0], discourse_lstm_states, self.dropout)

            if self.params.use_text_attention:
                final_text_states_c, final_text_states_h, final_text_state = self.get_text_attention(final_text_states_c, final_text_states_h, final_text_state, num_texts_to_keep)

            # state_positional_embeddings
            text_states, flat_seq = self._add_positional_embeddings(utter_hidden_states, utter_seqs)

            if len(prev_output) > 0:
                prev_outputs, prev_query_states = self.get_prev_queries(prev_outputs, prev_query_states, prev_output, schema)

            if len(gold_query) <= max_gen_length and len(prev_output) <= max_gen_length:
                loss, decoder_states = self.loss_turn(
                    final_text_state,
                    text_states,
                    schema_states,
                    max_gen_length,
                    gold_query=gold_query,
                    utter_seq=flat_seq,
                    prev_outputs=prev_outputs,
                    prev_query_states=prev_query_states,
                    schema=schema,
                    feed_gold_tokens=True,
                    training=True)
                total_gold_tokens += len(gold_query)
                losses.append(loss)
            else:
                # seq was too long to run the decoder
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
        utter_seqs = []

        final_text_states_c = []
        final_text_states_h = []

        prev_query_states = []
        prev_outputs = []

        discourse_state = None
        if self.params.discourse_level_lstm:
            discourse_state, discourse_lstm_states = self._initialize_discourse_states()
        discourse_states = []

        # Schema and schema embeddings
        schema = interaction.get_schema()
        schema_states = []

        if schema and not self.params.use_bert:
            schema_states = self.encode_schema_bow_simple(schema)

        interaction.start_interaction()
        while not interaction.done():
            text = interaction.next_text()

            prev_output = text.prev_output()

            utter_seq = text.utter_seq()

            if not self.params.use_bert:
                if self.params.discourse_level_lstm:
                    text_token_embedder = lambda token: torch.cat([self.utter_embedder(token), discourse_state], dim=0)
                else:
                    text_token_embedder = self.utter_embedder
                final_text_state, text_states = self.text_encoder(
                    utter_seq,
                    text_token_embedder)
            else:
                final_text_state, text_states, schema_states = self.get_bert_encoding(utter_seq, schema, discourse_state, dropout=False)

            utter_hidden_states.extend(text_states)
            utter_seqs.append(utter_seq)

            num_texts_to_keep = min(self.params.maximum_texts, len(utter_seqs))

            if self.params.discourse_level_lstm:
                _, discourse_state, discourse_lstm_states = torch_utils.forward_one_multilayer(self.discourse_lstms, final_text_state[1][0], discourse_lstm_states)

            if self.params.use_text_attention:
               final_text_states_c, final_text_states_h, final_text_state = self.get_text_attention(final_text_states_c, final_text_states_h, final_text_state, num_texts_to_keep)

            if self.params.state_positional_embeddings:
                text_states, flat_seq = self._add_positional_embeddings(utter_hidden_states, utter_seqs)
            else:
                flat_seq = []
                for utt in utter_seqs[-num_texts_to_keep:]:
                    flat_seq.extend(utt)

            if self.params.use_prev_query and len(prev_output) > 0:
                prev_outputs, prev_query_states = self.get_prev_queries(prev_outputs, prev_query_states, prev_output, schema)

            results = self.predict_turn(final_text_state,
                                        text_states,
                                        schema_states,
                                        max_gen_length,
                                        utter_seq=flat_seq,
                                        prev_outputs=prev_outputs,
                                        prev_query_states=prev_query_states,
                                        schema=schema)

            predicted_seq = results[0]
            predictions.append(results)

        return predictions

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        print("Loaded model from file " + filename)
