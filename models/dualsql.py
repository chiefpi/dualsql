"""Symmetric model for dual learning."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils.vocab import BOS_TOK, EOS_TOK, UNK_TOK

from models.modules.embedder import load_all_embs, Embedder
from models.modules.text_schema_encoder import TextSchemaEncoder
from models.modules.attention import Attention


class DualSQL(nn.Module):
    """Interaction model, where an interaction is processed all at once."""

    def __init__(
            self,
            utter_vocab,
            query_vocab,
            schema_vocab,
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

        super(DualSQL, self).__init__()
        # Embedders
        utter_vocab_emb, query_vocab_emb, schema_vocab_emb, glove_emb_size = load_all_embs(
            utter_vocab, query_vocab, schema_vocab, emb_filename)

        if use_bert: # TODO: bert
            pass
            # self.model_bert, self.tokenizer, self.bert_config = utils_bert.get_bert(params)
        else:
            self.utter_embedder = Embedder(
                emb_size,
                utter_vocab,
                init=utter_vocab_emb,
                freeze=freeze)

            self.column_name_token_embedder = Embedder(
                emb_size,
                schema_vocab,
                init=schema_vocab_emb,
                freeze=freeze)

        self.query_embedder = Embedder(
            emb_size,
            query_vocab,
            init=query_vocab_emb,
            freeze=False)

        # Positional embedder for inputs
        # if params.state_positional_embeddings:
        #     attention_key_size += params.positional_embedding_size
        #     self.positional_embedder = Embedder(
        #         params.positional_embedding_size,
        #         num_tokens=params.maximum_utters)

        # Encoders
        # encoder_input_size = self.bert_config.hidden_size if use_bert else emb_size
        encoder_input_size = emb_size
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
            dropout=dropout if encoder_num_layers > 1 else 0,
            bidirectional=True)

        self.utter_encoder = nn.LSTM(
            encoder_input_size,
            encoder_state_size//2,
            encoder_num_layers,
            dropout=dropout if encoder_num_layers > 1 else 0,
            bidirectional=True)

        self.query_encoder = nn.LSTM(
            encoder_input_size,
            encoder_state_size//2,
            encoder_num_layers,
            dropout=dropout if encoder_num_layers > 1 else 0,
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
        if use_editing: # TODO
            pass
            # self.query_decoder = SequencePredictorWithSchema(
            #     decoder_input_size,
            #     self.query_embedder,
            #     self.column_name_token_embedder,
            #     token_predictor)
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

        self.transform = nn.Linear(decoder_input_size, emb_size, bias=False)
        self.utt_matrix = nn.Linear(emb_size, len(utter_vocab))
        self.sql_matrix = nn.Linear(emb_size, len(query_vocab))
        self.col_matrix = nn.Linear(emb_size, emb_size, bias=False)

        self.emb_size = emb_size
        self.use_editing = use_editing
        self.max_turns_to_keep = max_turns_to_keep
        self.use_bert = use_bert
        self.decoder_num_layers = decoder_num_layers

    def forward(self, interaction, primal, max_gen_len, force=False):
        """Forwards on a single interaction.

        Args:
            primal (bool): Direction of forward, utterance-query/query-utterance.
            interaction (Interaction): Utterances, queries, and schema.
            max_gen_len (int): Maximum generation length.
            force (bool): Use teaching forcing.

        Returns:
            list of Tensor: Distributions over keywords and column names in every turn.
        """
        if primal:
            all_input_seqs = interaction.utter_seqs_id()
            all_output_seqs_gt = interaction.query_seqs_id()
            input_embedder = self.utter_embedder
            input_encoder = self.utter_encoder
            input_schema_encoder = self.utter_schema_encoder
            input_token_attention = self.utter_token_attention
            output_token_attention = self.query_token_attention
            output_embedder = self.query_embedder
            output_decoder = self.query_decoder
            output_matrix = self.sql_matrix
        else:
            all_input_seqs = interaction.query_seqs_id()
            all_output_seqs_gt = interaction.utter_seqs_id()
            input_embedder = self.query_embedder
            input_encoder = self.query_encoder
            input_schema_encoder = self.query_schema_encoder
            input_token_attention = self.query_token_attention
            output_token_attention = self.utter_token_attention
            output_embedder = self.utter_embedder
            output_decoder = self.utter_decoder
            output_matrix = self.utt_matrix
            
        schema = interaction.schema

        # History
        all_dist_seqs = []
        all_output_seqs = []
        prev_input_embs = []
        prev_output_embs = []
        prev_final_states_h = []
        prev_final_states_c = []

        # Embeds schema
        schema_embs = []
        if not self.use_bert:
            for column_name_sep in schema.schema_tokens_sep_id:
                sub_embs = self.column_name_token_embedder(column_name_sep).unsqueeze(1)
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
                input_embs = []
                for index in input_seq:
                    offset = len(input_embedder.vocab)
                    if index < offset:
                        input_emb = input_embedder([index]).unsqueeze(1)
                    else:
                        input_emb = schema_embs[index-offset].unsqueeze(0)
                    input_embs.append(input_emb)
                input_embs = torch.cat(input_embs, 0)
                input_embs, _ = input_encoder(input_embs)
                schema_embs, input_embs, final_input_state = input_schema_encoder(schema_embs, input_embs)

            # 1 x batch_size x state_size
            final_input_state_h = final_input_state[0].transpose(0, 1).contiguous().view(1, -1, self.emb_size)
            final_input_state_c = final_input_state[1].transpose(0, 1).contiguous().view(1, -1, self.emb_size)
            init_decoder_state_h = final_input_state_h
            init_decoder_state_c = final_input_state_c
            if i > 0:
                history_states_h = torch.cat(prev_final_states_h, 0)
                history_states_c = torch.cat(prev_final_states_c, 0)
                init_decoder_state_h = init_decoder_state_h + self.turn_attention(final_input_state_h, history_states_h)
                init_decoder_state_c = init_decoder_state_h + self.turn_attention(final_input_state_c, history_states_c)
            # num_layers x batch_size x state_size
            init_decoder_state = (
                init_decoder_state_h.repeat(self.decoder_num_layers, 1, 1),
                init_decoder_state_c.repeat(self.decoder_num_layers, 1, 1))

            prev_input_embs.append(input_embs)
            prev_final_states_h.append(final_input_state_h)
            prev_final_states_c.append(final_input_state_c)
            prev_input_embs = prev_input_embs[:self.max_turns_to_keep]
            prev_final_states_h = prev_final_states_h[:self.max_turns_to_keep]
            prev_final_states_c = prev_final_states_c[:self.max_turns_to_keep]

            decoder_hidden = init_decoder_state_h
            decoder_state = init_decoder_state
            bos_id = output_embedder.vocab.token2id[BOS_TOK]
            # eos_id = output_embedder.vocab.token2id[EOS_TOK]
            output_emb = output_embedder([bos_id]).unsqueeze(1) # 1 x batch_size x emb_size
            output_embs = [output_emb]
            dist_seqs = []
            output_seq = [BOS_TOK]

            offset = len(output_embedder.vocab)
            output_seq_gt = all_output_seqs_gt[i]
            for j in range(min(len(output_seq_gt), max_gen_len) - 1):
                if force:
                    index = output_seq_gt[j]
                    if index < offset:
                        output_emb = output_embedder([index]).unsqueeze(1)
                        output_token = output_embedder.vocab.id2token[index]
                    else:
                        output_emb = schema_embs[index-offset].unsqueeze(0)
                        output_token = schema.vocab.id2token[index-offset]

                # 1 x batch_size x emb_size
                context_schema = self.schema_attention(decoder_hidden, schema_embs)
                context_input = input_token_attention(decoder_hidden, torch.cat(prev_input_embs, 0).view(-1, 1, self.emb_size))
                if i > 0:
                    context_output = output_token_attention(decoder_hidden, torch.cat(prev_output_embs, 0).view(-1, 1, self.emb_size))
                else:
                    context_output = torch.zeros_like(context_input)
                # 1 x batch_size x 3*emb_size
                context = torch.cat([context_schema, context_input, context_output], -1)

                decoder_hidden, decoder_state = output_decoder(
                    torch.cat([output_emb, context], -1), decoder_state)

                output = torch.tanh(self.transform(torch.cat([decoder_hidden, context], -1)))
                score = output_matrix(output)
                if primal:
                    score_col = torch.bmm(self.col_matrix(output).transpose(0, 1), schema_embs.transpose(0, 1).transpose(1, 2))
                    score = torch.cat([score, score_col], -1)

                dist = F.log_softmax(score, -1)
                dist_seqs.append(dist)

                argmax_id = torch.argmax(dist).item()
                if argmax_id < offset:
                    output_emb = output_embedder([argmax_id]).unsqueeze(1)
                    output_token = output_embedder.vocab.id2token[argmax_id]
                else:
                    output_emb = schema_embs[argmax_id-offset].unsqueeze(0)
                    output_token = schema.vocab.id2token[argmax_id-offset]
                output_embs.append(output_emb)
                output_seq.append(output_token)

                # if not force and argmax_id == eos_id:
                #     break

            prev_output_embs.append(torch.cat(output_embs, 0)) # len x batch_size x emb_dim
            prev_output_embs = prev_output_embs[:self.max_turns_to_keep]
            all_dist_seqs.append(torch.cat(dist_seqs, 0))
            all_output_seqs.append(output_seq)

        return all_dist_seqs, all_output_seqs

    def forward_seqs(self, all_input_seqs, schema, primal, max_gen_len):
        """Forwards on sequences.

        Args:
            primal (bool): Direction of forward, utterance-query/query-utterance.
            max_gen_len (int): Maximum generation length.
            force (bool): Use teaching forcing.

        Returns:
            list of Tensor: Distributions over keywords and column names in every turn.
        """
        if primal:
            input_embedder = self.utter_embedder
            input_encoder = self.utter_encoder
            input_schema_encoder = self.utter_schema_encoder
            input_token_attention = self.utter_token_attention
            output_token_attention = self.query_token_attention
            output_embedder = self.query_embedder
            output_decoder = self.query_decoder
            output_matrix = self.sql_matrix
        else:
            input_embedder = self.query_embedder
            input_encoder = self.query_encoder
            input_schema_encoder = self.query_schema_encoder
            input_token_attention = self.query_token_attention
            output_token_attention = self.utter_token_attention
            output_embedder = self.utter_embedder
            output_decoder = self.utter_decoder
            output_matrix = self.utt_matrix
            
        # History
        all_dist_seqs = []
        all_output_seqs = []
        all_output_seqs_id = []
        
        prev_input_embs = []
        prev_output_embs = []
        prev_final_states_h = []
        prev_final_states_c = []

        # Embeds schema
        schema_embs = []
        if not self.use_bert:
            for column_name_sep in schema.schema_tokens_sep_id:
                sub_embs = self.column_name_token_embedder(column_name_sep).unsqueeze(1)
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
                input_embs = []
                for index in input_seq:
                    offset = len(input_embedder.vocab)
                    if index < offset:
                        input_emb = input_embedder([index]).unsqueeze(1)
                    else:
                        input_emb = schema_embs[index-offset].unsqueeze(0)
                    input_embs.append(input_emb)
                input_embs = torch.cat(input_embs, 0)
                input_embs, _ = input_encoder(input_embs)
                schema_embs, input_embs, final_input_state = input_schema_encoder(schema_embs, input_embs)

            # 1 x batch_size x state_size
            final_input_state_h = final_input_state[0].transpose(0, 1).contiguous().view(1, -1, self.emb_size)
            final_input_state_c = final_input_state[1].transpose(0, 1).contiguous().view(1, -1, self.emb_size)
            init_decoder_state_h = final_input_state_h
            init_decoder_state_c = final_input_state_c
            if i > 0:
                history_states_h = torch.cat(prev_final_states_h, 0)
                history_states_c = torch.cat(prev_final_states_c, 0)
                init_decoder_state_h = init_decoder_state_h + self.turn_attention(final_input_state_h, history_states_h)
                init_decoder_state_c = init_decoder_state_h + self.turn_attention(final_input_state_c, history_states_c)
            # num_layers x batch_size x state_size
            init_decoder_state = (
                init_decoder_state_h.repeat(self.decoder_num_layers, 1, 1),
                init_decoder_state_c.repeat(self.decoder_num_layers, 1, 1))

            prev_input_embs.append(input_embs)
            prev_final_states_h.append(final_input_state_h)
            prev_final_states_c.append(final_input_state_c)
            prev_input_embs = prev_input_embs[:self.max_turns_to_keep]
            prev_final_states_h = prev_final_states_h[:self.max_turns_to_keep]
            prev_final_states_c = prev_final_states_c[:self.max_turns_to_keep]

            decoder_hidden = init_decoder_state_h
            decoder_state = init_decoder_state
            bos_id = output_embedder.vocab.token2id[BOS_TOK]
            eos_id = output_embedder.vocab.token2id[EOS_TOK]
            output_emb = output_embedder([bos_id]).unsqueeze(1) # 1 x batch_size x emb_size
            output_embs = [output_emb]
            dist_seqs = []
            output_seq = [BOS_TOK]

            offset = len(output_embedder.vocab)
            for j in range(max_gen_len - 1):
                # 1 x batch_size x emb_size
                context_schema = self.schema_attention(decoder_hidden, schema_embs)
                context_input = input_token_attention(decoder_hidden, torch.cat(prev_input_embs, 0).view(-1, 1, self.emb_size))
                if i > 0:
                    context_output = output_token_attention(decoder_hidden, torch.cat(prev_output_embs, 0).view(-1, 1, self.emb_size))
                else:
                    context_output = torch.zeros_like(context_input)
                # 1 x batch_size x 3*emb_size
                context = torch.cat([context_schema, context_input, context_output], -1)

                decoder_hidden, decoder_state = output_decoder(
                    torch.cat([output_emb, context], -1), decoder_state)

                output = torch.tanh(self.transform(torch.cat([decoder_hidden, context], -1)))
                score = output_matrix(output)
                if primal:
                    score_col = torch.bmm(self.col_matrix(output).transpose(0, 1), schema_embs.transpose(0, 1).transpose(1, 2))
                    score = torch.cat([score, score_col], -1)

                dist = F.log_softmax(score, -1)
                dist_seqs.append(dist)

                argmax_id = torch.argmax(dist).item()
                if argmax_id < offset:
                    output_emb = output_embedder([argmax_id]).unsqueeze(1)
                    output_token = output_embedder.vocab.id2token[argmax_id]
                else:
                    output_emb = schema_embs[argmax_id-offset].unsqueeze(0)
                    output_token = schema.vocab.id2token[argmax_id-offset]
                output_embs.append(output_emb)
                output_seq.append(output_token)

                if argmax_id == eos_id:
                    break

            prev_output_embs.append(torch.cat(output_embs, 0)) # len x batch_size x emb_dim
            prev_output_embs = prev_output_embs[:self.max_turns_to_keep]
            all_dist_seqs.append(torch.cat(dist_seqs, 0))
            all_output_seqs.append(output_seq)

        return all_dist_seqs, all_output_seqs, all_output_seqs_id

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        print("Loaded model from file " + filename)
