"""Symmetric model for dual learning."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils.vocab import BOS_TOK, EOS_TOK, UNK_TOK

from modules.embedder import load_all_embs, Embedder
from modules.text_schema_encoder import TextSchemaEncoder
from modules.attention import Attention
from modules.decoder import SequencePredictorWithSchema
from modules.token_predictor import SchemaTokenPredictor


class DualSQL(nn.Module):
    """Interaction model, where an interaction is processed all at once."""

    def __init__(
            self,
            utter_vocab,
            query_vocab,
            query_vocab_schema,
            emb_filename,
            emb_size=300,
            encoder_state_size=300,
            encoder_num_layers=1,
            decoder_state_size=300,
            decoder_num_layers=2,
            dropout=0,
            freeze=False,
            use_editing=False,
            max_turns_to_keep=5,
            use_bert=False):

        # Embedders
        utter_vocab_emb, query_vocab_emb, query_schema_vocab_emb, utter_emb_size = load_all_embs(
            utter_vocab, query_vocab, query_vocab_schema, emb_filename)

        if use_bert: # TODO: bert
            pass
            # self.model_bert, self.tokenizer, self.bert_config = utils_bert.get_bert(params)
        else:
            self.utter_embedder = Embedder(
                utter_emb_size, # 300
                init=utter_vocab_emb,
                vocab=utter_vocab,
                freeze=freeze)

            self.column_name_token_embedder = Embedder(
                emb_size,
                init=query_schema_vocab_emb,
                vocab=query_vocab_schema,
                freeze=freeze)

        self.query_embedder = Embedder(
            emb_size,
            init=query_vocab_emb,
            vocab=query_vocab,
            freeze=False)

        # Positional embedder for inputs
        # if params.state_positional_embeddings:
        #     attention_key_size += params.positional_embedding_size
        #     self.positional_embedder = Embedder(
        #         params.positional_embedding_size,
        #         num_tokens=params.maximum_utters)

        # Encoders
        encoder_input_size = self.bert_config.hidden_size if use_bert else emb_size
        # encoder_input_size += encoder_state_size//2 # discourse state

        # self.discourse_encoder = nn.LSTM(
        #     encoder_state_size,
        #     encoder_state_size//2)
        # Trainable initial state
        # self.discourse_state = nn.Parameter(torch.empty(encoder_state_size/2).uniform_(-0.1, 0.1))
        # attention_key_size = encoder_state_size

        # First layer Bi-LSTMs
        self.schema_encoder = nn.LSTM(
            emb_size,
            encoder_state_size//2,
            encoder_num_layers,
            dropout=dropout,
            bidirectional=True)

        self.utter_encoder = nn.LSTM(
            encoder_input_size,
            encoder_state_size//2,
            encoder_num_layers,
            dropout=dropout,
            bidirectional=True)

        self.query_encoder = nn.LSTM(
            encoder_input_size,
            encoder_state_size//2,
            encoder_num_layers,
            dropout=dropout,
            bidirectional=True)

        # Co-attention encoder
        self.utter_schema_encoder = TextSchemaEncoder(
            emb_size,
            encoder_num_layers,
            dropout=dropout)

        self.query_schema_encoder = TextSchemaEncoder(
            emb_size,
            encoder_num_layers,
            dropout=dropout)

        # Turn attention
        self.turn_attention = Attention(emb_size, emb_size)

        # Dual attention
        self.schema_attention = Attention(decoder_state_size, emb_size)
        self.utter_token_attention = Attention(decoder_state_size, emb_size)
        self.query_token_attention = Attention(decoder_state_size, emb_size)

        # Decoders
        # token_predictor = SchemaTokenPredictor(
        #     params,
        #     query_vocab,
        #     attention_key_size*2,
        #     attention_key_size*2)
        # Use schema_attention in decoder
        decoder_input_size = emb_size * 4
        if use_editing:
            self.query_decoder = SequencePredictorWithSchema(
                decoder_input_size,
                self.query_embedder,
                self.column_name_token_embedder,
                token_predictor)
        else:
            self.query_decoder = nn.LSTM(
                decoder_input_size,
                decoder_state_size,
                decoder_num_layers,
                dropout=dropout)

        self.utter_decoder = nn.LSTM(
            decoder_input_size,
            decoder_state_size,
            decoder_num_layers,
            dropout=dropout)

        self.emb_size = emb_size
        self.use_editing = use_editing
        self.max_turns_to_keep = max_turns_to_keep
        self.use_bert = use_bert

    def forward(
            self,
            primal,
            interaction,
            max_gen_len):
        """Forwards on a single interaction.

        Args:
            primal (bool): Direction of forward, utterance-query/query-utterance.
            interaction (Interaction)
            max_gen_len (int): Maximum generation length.

        Returns:
            distribution?
        """
        if primal:
            all_input_seqs = interaction.utter_seqs()
            input_embedder = self.utter_embedder            
            input_encoder = self.utter_encoder
            input_token_attention = self.utter_token_attention
            output_token_attention = self.query_token_attention
            output_embbder = self.query_embedder
            output_decoder = self.query_decoder
        else:
            all_input_seqs = interaction.query_seqs()
            input_embedder = self.query_embedder
            input_encoder = self.query_encoder
            input_token_attention = self.query_token_attention
            output_token_attention = self.utter_token_attention
            output_embbder = self.utter_embedder
            output_decoder = self.utter_decoder
            
        schema = interaction.schema

        # History
        prev_input_seqs = []
        prev_input_embs = [] # TODO: padding
        prev_output_seqs = []
        prev_output_embs = []
        prev_final_states = []
        decoder_states = []

        # Embeds schema
        schema_embs = []
        if not self.use_bert:
            for column_name in schema.column_names_embedder_input:
                sub_embs = self.column_name_token_embedder(column_name.split()).unsqueeze(1)
                _, (column_name_emb, _) = self.schema_encoder(sub_embs)
                schema_embs.append(column_name_emb.view(1, -1))
            # schema_len x batch_size x state_size
            schema_embs = torch.stack(schema_embs, 0)

        # discourse_state = self.init_discourse_state()
        for i, input_seq in enumerate(all_input_seqs):
            # Embeds schema and input with co-attention
            if self.use_bert: # TODO: bert
                pass
                # last_input_state, input_embs, schema_embs = self.get_bert_encoding(
                #     input_seq, schema_embs, discourse_state, dropout=True)
            else:
                input_embs = input_embedder(input_seq).unsqueeze(1)
                # input_embs = torch.cat([input_embs, discourse_state], -1)
                input_embs, _ = input_encoder(input_embs)
                schema_embs, input_embs, final_input_state = self.text_schema_encoder(schema_embs, input_embs)

            num_turns_to_keep = min(self.max_turns_to_keep, i+1)

            # 1 x batch_size x state_size
            init_decoder_state_h = final_input_state[0].transpose(0, 1).contiguous().view(1, -1, self.emb_size)
            init_decoder_state_c = final_input_state[1].transpose(0, 1).contiguous().view(1, -1, self.emb_size)
            if i > 0:
                history_states = torch.cat(prev_final_states, 0)
                init_decoder_state_h += self.turn_attention(init_decoder_state_h, history_states)
                init_decoder_state_c += self.turn_attention(init_decoder_state_c, history_states)
            init_decoder_state = (init_decoder_state_h.repeat(2, 1, 1), init_decoder_state_c.repeat(2, 1, 1))

            prev_input_seqs.append(input_seq)
            # prev_input_embs.append(input_embs)
            prev_final_states.append(final_input_state)

            # if self.params.use_state_positional_embedding:
            #     input_encodings, flat_seq = self._add_positional_embeddings(
            #         input_encodings, prev_input_seqs)
            # else:
            #     flat_seq = []
            #     for utt in prev_input_seqs[-num_turns_to_keep:]:
            #         flat_seq.extend(utt)

            # Decoding
            # if primal:
            #     decoder_output, _ = self.query_decoder(
            #         input_embs,
            #         prev_input_embs,
            #         schema_embs,
            #         max_gen_len,
            #         input_seq=input_seq,
            #         prev_output_seqs=prev_output_seqs,
            #         prev_output_states=prev_output_states,
            #         schema=schema)
            # else:
            #     decoder_output, decoder_state = self.utter_decoder(
            #         torch.cat([output_embs, context], -1),
            #         decoder_state)

            continue_gen = True
            decoder_state = init_decoder_state
            output_emb = output_embbder(BOS_TOK)
            while continue_gen:
                # decoder_state_size x batch_size x emb_size
                context_schema = self.schema_attention(decoder_state, schema_embs)
                context_input = input_token_attention(decoder_state, torch.cat(prev_input_embs, 1).view(-1, 1, self.emb_size))
                if i > 0:
                    context_output = output_token_attention(decoder_state, torch.cat(prev_output_embs, 1).view(-1, 1, self.emb_size))
                # decoder_state_size x batch_size x 3*emb_size
                context = torch.cat([context_schema, context_input, context_output], -1)

                decoder_output, decoder_state = output_decoder(
                    torch.cat([output_emb, context], -1),
                    decoder_state)

                if j == max_gen_len or argmax_token == EOS_TOK:
                    continue_gen = False

            all_scores = []
            all_alignments = []
            for prediction in decoder_results.predictions:
                scores = F.softmax(prediction.scores, 0)
                alignments = prediction.aligned_tokens
                if i > 0:
                    query_scores = F.softmax(prediction.query_scores, 0)
                    copy_switch = prediction.copy_switch
                    scores = torch.cat([scores*(1-copy_switch), query_scores*copy_switch], 0)
                    alignments = alignments + prediction.query_tokens

                all_scores.append(scores)
                all_alignments.append(alignments)
                
            predicted_seq = decoder_results.seq
            fed_seq = predicted_seq
                
            decoder_states = [pred.decoder_state for pred in decoder_results.predictions]

            if self.use_editing:
                prev_output_seqs = prev_output_seqs[-num_turns_to_keep:]
                output_token_embedder = lambda output_token: self.get_output_token_embedding(output_token, schema)
                _, new_predicted_seq_states = self.query_encoder(predicted_seq, output_token_embedder)
                assert len(new_predicted_seq_states) == len(predicted_seq)
                prev_output_states.append(new_predicted_seq_states)
                prev_output_states = prev_output_states[-num_turns_to_keep:]
                
        return predicted_seq, decoder_states, decoder_results


    def loss(self, interaction, max_gen_len, snippet_align_prob=1.):
        """Calculates the loss on a single interaction.

        Args:
            interaction (InteractionItem): An interaction to train on.
            max_gen_len (int): Maximum generation length.

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

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        print("Loaded model from file " + filename)
