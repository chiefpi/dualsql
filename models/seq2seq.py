"""Baseline"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2Seq(nn.Module):
    """Contains Seq2Seq with attention.
    Encoder: Bi-LSTM
    Decoder: undirectional LSTM
    """
    def __init__(
            self,
            src_vocab_size,
            tgt_vocab_size,
            src_unk_idx=1,
            tgt_unk_idx=1,
            pad_src_idxs=[0],
            pad_tgt_idxs=[0],
            src_emb_dim=300,
            tgt_emb_dim=300,
            hidden_dim=300,
            num_layers=1,
            dropout=0.5):
        super(Seq2Seq, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p=dropout)
        self.src_embedder = nn.Embedding(src_vocab_size, src_emb_dim)
        self.tgt_embedder = nn.Embedding(tgt_vocab_size, tgt_emb_dim)

        self.encoder = nn.LSTM(
            src_emb_dim,
            hidden_dim,
            num_layers,
            bidirectional=True, 
            dropout=dropout)

        self.Wa = nn.Linear(hidden_dim*2, hidden_dim, bias=False)
        self.Va = nn.Linear(hidden_dim, 1, bias=False)

        self.decoder = nn.LSTM(
            tgt_emb_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
            dropout=dropout)
        self.affine = nn.Linear(hidden_dim * 2, tgt_emb_dim)
        self.generator = nn.Linear(tgt_emb_dim, tgt_vocab_size)

    # def init_param(self):
    #     for p in model.parameters():
    #         p.data.uniform_(-init, init)
    #     for pad_token_idx in pad_src_idxs:
    #         model.src_embedder.embed.weight.data[pad_token_idx].zero_()
    #     for pad_token_idx in pad_tgt_idxs:
    #         model.tgt_embedder.embed.weight.data[pad_token_idx].zero_()

    def forward(self, src_inputs, src_lens, tgt_inputs):
        # Mask out unknown tokens
        token_mask = src_inputs >= self.src_vocab_size
        if token_mask.any():
            src_inputs = src_inputs.masked_fill_(token_mask, self.unk_idx)
        token_mask = tgt_inputs >= self.tgt_vocab_size
        if token_mask.any():
            tgt_inputs = tgt_inputs.masked_fill_(token_mask, self.unk_idx)

        src_emb = self.dropout(self.src_embedder(src_inputs))
        enc_out, hidden_states = self.encoder(src_emb, src_lens)
        
        enc_h, enc_c = hidden_states
        dec_h = enc_h.new_zeros(self.num_layers, enc_h.size(1), enc_h.size(2))
        dec_c = enc_c.new_zeros(self.num_layers, enc_c.size(1), enc_c.size(2))
        hidden_states = (dec_h, dec_c)
        
        # src_mask = lens2mask(src_lens)

        tgt_emb = self.dropout(self.tgt_embedder(tgt_inputs))
        dec_out, hidden_states = self.decoder(tgt_emb, hidden_states)
        context = []
        for i in range(dec_out.size(1)):
            # tmp_context, _ = self.attn(enc_out, dec_out[:, i, :], src_mask)
            d = dec_out[:, i, :].unsqueeze(dim=1).repeat(1, enc_out.size(1), 1)
            e = self.Wa(torch.cat([d, enc_out], dim=-1))
            e = self.Va(torch.tanh(e)).squeeze(dim=-1)
            e.masked_fill_(src_mask == 0, -math.inf)
            a = F.softmax(e, dim=1)
            tmp_context = torch.bmm(a.unsqueeze(1), enc_out)
            context.append(tmp_context)

        context = torch.cat(context, dim=1)
        feats = torch.cat([dec_out, context], dim=-1)
        feats = self.affine(self.dropout(feats))

        out = F.log_softmax(self.generator(self.dropout(feats)), dim=-1)

        return out

    def decode_one_step(self, ys, hidden_states, memory, src_mask):
        """
        @ys: bsize x 1
        """
        # dec_out, _ = self.decoder(self.tgt_embedder(ys), hidden_states, memory, src_mask, copy_tokens)
        dec_out, hidden_states = self.decoder(self.tgt_embedder(ys), hidden_states)
        context = []
        for i in range(dec_out.size(1)):
            # tmp_context, _ = self.attn(memory, dec_out[:, i, :], src_mask)
            d = dec_out[:, i, :].unsqueeze(dim=1).repeat(1, memory.size(1), 1)
            e = self.Wa(torch.cat([d, memory], dim=-1))
            e = self.Va(torch.tanh(e)).squeeze(dim=-1)
            e.masked_fill_(src_mask == 0, -math.inf)
            a = F.softmax(e, dim=1)
            tmp_context = torch.bmm(a.unsqueeze(1), memory)
            context.append(tmp_context)
        context = torch.cat(context, dim=1)
        feats = torch.cat([dec_out, context], dim=-1)
        feats = self.affine(feats)

        # out = self.generator(dec_out)
        out = F.log_softmax(self.generator(feats), dim=-1)

        return out.squeeze(dim=1), hidden_states

    def pad_embedding_grad_zero(self):
        self.src_embedder.pad_embedding_grad_zero()
        self.tgt_embedder.pad_embedding_grad_zero()

    def load_model(self, load_dir):
        self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))