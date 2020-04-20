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

        utter_emb_size = params.utter_emb_size
        query_emb_size = params.query_emb_size
        encoder_state_size = params.encoder_state_size
        encoder_num_layers = params.encoder_num_layers
        schema_encoder_input_size = params.input_embedding_size
        schema_encoder_state_size = params.encoder_state_size
        schema_encoder_num_layers = 1
        dropout = params.dropout
        freeze = params.freeze
        self.max_turns_to_keep = params.max_turns_to_keep

        # Embedders
        utter_vocab_emb, query_vocab_emb, query_schema_vocab_emb, utter_emb_size = load_all_embs(
            utter_vocab, query_vocab, query_vocab_schema, params.embedding_filename)

        if params.use_bert: # TODO: bert
            self.model_bert, self.tokenizer, self.bert_config = utils_bert.get_bert(params)
        else:
            self.utter_embedder = Embedder(
                utter_emb_size,
                init=utter_vocab_emb,
                vocab=utter_vocab,
                freeze=freeze)

            self.column_name_token_embedder = Embedder(
                utter_emb_size,
                init=query_schema_vocab_emb,
                vocab=query_vocab_schema,
                freeze=freeze)

        self.query_embedder = Embedder(
            query_emb_size,
            init=query_vocab_emb,
            vocab=query_vocab,
            freeze=False)

        # Positional embedder for inputs
        attention_key_size = encoder_state_size
        schema_attention_key_size = attention_key_size
        # if params.state_positional_embeddings:
        #     attention_key_size += params.positional_embedding_size
        #     self.positional_embedder = Embedder(
        #         params.positional_embedding_size,
        #         num_tokens=params.maximum_utters)
        text_attention_key_size = attention_key_size

        # Encoders
        encoder_input_size = self.bert_config.hidden_size if params.use_bert else params.input_emb_size
        encoder_input_size += encoder_state_size / 2 # discourse-level lstm

        self.discourse_encoder = nn.LSTM(
            encoder_state_size,
            encoder_state_size/2,
            dropout=dropout)

        self.text_schema_encoder = TextSchemaEncoder(
            schema_encoder_input_size,
            schema_encoder_state_size,
            schema_encoder_num_layers,
            schema_attention_key_size,
            encoder_input_size,
            encoder_state_size,
            encoder_num_layers,
            text_attention_key_size,
            dropout=dropout)

        # Decoders
        token_predictor = SchemaTokenPredictor(
            params,
            query_vocab,
            text_attention_key_size,
            schema_attention_key_size)

        # Use schema_attention in decoder
        decoder_input_size = query_emb_size + \
            text_attention_key_size + \
            schema_attention_key_size + \
            encoder_state_size
        self.query_decoder = SequencePredictorWithSchema(
            params,
            decoder_input_size,
            self.query_embedder,
            self.column_name_token_embedder,
            token_predictor)
        self.utter_decoder = nn.LSTM(
            decoder_input_size,
            decoder)

        self.params = params # TODO: remove

    def forward(
            self,
            primal,
            interaction,
            max_gen_length):
        """Forwards on a single interaction.

        Args:
            primal (bool): Direction of forward, utterance-query/query-utterance.
            interaction (Interaction)
            max_gen_length (int): Maximum generation length.

        Returns:
            distribution?
        """
        schema = interaction.schema
        all_input_seqs = interaction.utter_seqs() if primal else interaction.query_seqs()

        # History
        input_seqs = []
        input_states = []
        prev_output_seqs = []
        prev_output_states = []
        decoder_states = []

        # Schema embeddings
        schema_states = []
        if not self.params.use_bert:
            for column_name in schema.column_names_embedder_input:
                sub_embs = [self.column_name_token_embedder(token) for token in column_name.split()]
                column_name_emb = torch.stack(sub_embs, dim=0).mean(dim=0)
                schema_states.append(column_name_emb)

        discourse_state = nn.Parameter(torch.empty().uniform_(-0.1, 0.1))

        for i, input_seq in enumerate(all_input_seqs):
            # Schema and input embeddings
            if self.params.use_bert: # TODO: bert works for sql?
                final_input_state, new_input_states, schema_states = self.get_bert_encoding(
                    input_seq, schema_states, discourse_state, dropout=True)
            else:
                input_token_embedder = lambda token: torch.cat( # TODO: WTH discourse
                    [self.input_embedder(token), discourse_state], dim=0)
                if primal:
                    final_input_state, new_input_states = self.utter_encoder(
                        input_seq, input_token_embedder)
                else:
                    final_input_state, new_input_states = self.query_encoder(
                        input_seq, input_token_embedder)
                final_input_state = self.text_schema_encoder(schema_states, input_states)
                        

            input_seqs.append(input_seq)
            input_states.extend(new_input_states)

            num_turns_to_keep = min(self.max_turns_to_keep, i+1)

            # final_input_state[1][0] is the first layer's hidden states at the last time step (concat forward lstm and backward lstm)
            _, discourse_state = self.discourse_encoder(
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
                    prev_output_seqs=prev_output_seqs,
                    prev_output_states=prev_output_states,
                    schema=schema)
            else:
                decoder_results = self.utter_decoder(
                    final_input_state,
                    input_states,
                    schema_states,
                    max_gen_length,
                    input_seq=input_seq,
                    prev_output_states=prev_output_states,
                    schema=schema)
            predicted_seq = decoder_results.seq
            fed_seq = predicted_seq
                
            decoder_states = [pred.decoder_state for pred in decoder_results.predictions]

            if self.params.use_editing:
                prev_output_seqs = prev_output_seqs[-num_turns_to_keep:]
                output_token_embedder = lambda output_token: self.get_output_token_embedding(output_token, schema)
                _, new_predicted_seq_states = self.query_encoder(predicted_seq, output_token_embedder)
                assert len(new_predicted_seq_states) == len(predicted_seq)
                prev_output_states.append(new_predicted_seq_states)
                prev_output_states = prev_output_states[-num_turns_to_keep:]
                
            torch.cuda.empty_cache()

        return predicted_seq, decoder_states, decoder_results


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
                "Expected utterance and state sequence length to be the same, " \
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

        """


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

        prev_output_states = []
        prev_output_seqs = []

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
                prev_output_seqs, prev_output_states = self.get_prev_queries(prev_output_seqs, prev_output_states, prev_output, schema)

            results = self.predict_turn(final_text_state,
                                        text_states,
                                        schema_states,
                                        max_gen_length,
                                        utter_seq=flat_seq,
                                        prev_output_seqs=prev_output_seqs,
                                        prev_output_states=prev_output_states,
                                        schema=schema)

            predicted_seq = results[0]
            predictions.append(results)

        return predictions

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        print("Loaded model from file " + filename)
