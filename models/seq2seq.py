# TODO: remove it

import torch.nn as nn
import torch.nn.functional as F

from utils.constants import BOS, EOS, MAX_DECODE_LENGTH
from utils.tensors import tile, lens2mask
from beam import Beam, GNMTGlobalScorer


class Seq2Seq(nn.Module):
    """ Seq2Seq with attention
    Encoder: Bi-LSTM
    Decoder: undirectional LSTM
    """
    def __init__(
        self, src_vocab=None, tgt_vocab=None,
        src_unk_idx=1, tgt_unk_idx=1, pad_src_idxs=[0], pad_tgt_idxs=[0],
        src_emb_dim=100, tgt_emb_dim=100, hidden_dim=200, num_layers=1,
        bidirectional=True, dropout=0.5, init=None
    ):

        super(Seq2Seq, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.num_layers = num_layers

        # self.src_embed = RNNEmbeddings(src_emb_dim, src_vocab, src_unk_idx, pad_token_idxs=pad_src_idxs, dropout=dropout)
        self.src_embed = nn.Sequential(
            nn.Embedding(src_vocab, src_emb_dim),
            nn.Dropout(p=dropout)
        )
        # self.tgt_embed = RNNEmbeddings(tgt_emb_dim, tgt_vocab, tgt_unk_idx, pad_token_idxs=pad_tgt_idxs, dropout=dropout)
        self.tgt_embed = nn.Sequential(
            nn.Embedding(tgt_vocab, tgt_emb_dim),
            nn.Dropout(p=dropout)
        )
        # self.encoder = LSTM(src_emb_dim, hidden_dim, num_layers, cell=cell, bidirectional=bidirectional, dropout=dropout)
        self.encoder = nn.LSTM(
            src_emb_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional, 
            batch_first=True,
            dropout=dropout
        )
        # self.enc2dec = StateTransition(num_layers, cell=cell, bidirectional=bidirectional, hidden_dim=hidden_dim)
        # attn_model = Attention(hidden_dim * num_directions, hidden_dim)
        self.Wa = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.Va = nn.Linear(hidden_dim, 1, bias=False)
        # self.decoder = RNNDecoder(tgt_emb_dim, hidden_dim, num_layers, attn=attn_model, cell=cell, dropout=dropout)
        self.decoder = nn.LSTM(
            tgt_emb_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
            dropout=dropout
        )
        self.affine = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim * 2, tgt_emb_dim)
        )
        # self.generator = Generator(tgt_emb_dim, tgt_vocab, dropout=dropout)
        self.generator = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(tgt_emb_dim, tgt_vocab) # vocab vs dim?
        )


    # def init_param(self):
    #     for p in model.parameters():
    #         p.data.uniform_(-init, init)
    #     for pad_token_idx in pad_src_idxs:
    #         model.src_embed.embed.weight.data[pad_token_idx].zero_()
    #     for pad_token_idx in pad_tgt_idxs:
    #         model.tgt_embed.embed.weight.data[pad_token_idx].zero_()

    def forward(self, src_inputs, src_lens, tgt_inputs):
        # mask out unknown tokens
        token_mask = src_inputs >= self.src_vocab
        if token_mask.any():
            src_inputs = src_inputs.masked_fill_(token_mask, self.unk_idx)
        token_mask = tgt_inputs >= self.tgt_vocab
        if token_mask.any():
            tgt_inputs = tgt_inputs.masked_fill_(token_mask, self.unk_idx)

        # enc_out, hidden_states = self.encoder(self.src_embed(src_inputs), src_lens)
        enc_out, hidden_states = self.encoder(self.src_embed(src_inputs), src_lens)
        
        # hidden_states = self.enc2dec(hidden_states)
        enc_h, enc_c = hidden_states
        dec_h = enc_h.new_zeros(self.num_layers, enc_h.size(1), enc_h.size(2))
        dec_c = enc_c.new_zeros(self.num_layers, enc_c.size(1), enc_c.size(2))
        hidden_states = (dec_h, dec_c)
        
        src_mask = lens2mask(src_lens)

        # dec_out, _ = self.decoder(self.tgt_embed(tgt_inputs), hidden_states, enc_out, src_mask, copy_tokens)
        dec_out, hidden_states = self.decoder(self.tgt_embed(tgt_inputs), hidden_states)
        context = []
        for i in range(dec_out.size(1)):
            # tmp_context, _ = self.attn(enc_out, dec_out[:, i, :], src_mask)
            d = dec_out[:, i, :].unsqueeze(dim=1).repeat(1, enc_out.size(1), 1)
            e = self.Wa(torch.cat([d, enc_out], dim=-1))
            e = self.Va(torch.tanh(e)).squeeze(dim=-1)
            e.masked_fill_(src_mask == 0, -float('inf'))
            a = F.softmax(e, dim=1)
            tmp_context = torch.bmm(a.unsqueeze(1), enc_out)
            context.append(tmp_context)
        context = torch.cat(context, dim=1)
        feats = torch.cat([dec_out, context], dim=-1)
        feats = self.affine(feats)

        # out = self.generator(dec_out)
        out = F.log_softmax(self.generator(feats), dim=-1)

        return out

    def decode_one_step(self, ys, hidden_states, memory, src_mask):
        """
        @ys: bsize x 1
        """
        # dec_out, _ = self.decoder(self.tgt_embed(ys), hidden_states, memory, src_mask, copy_tokens)
        dec_out, hidden_states = self.decoder(self.tgt_embed(ys), hidden_states)
        context = []
        for i in range(dec_out.size(1)):
            # tmp_context, _ = self.attn(memory, dec_out[:, i, :], src_mask)
            d = dec_out[:, i, :].unsqueeze(dim=1).repeat(1, memory.size(1), 1)
            e = self.Wa(torch.cat([d, memory], dim=-1))
            e = self.Va(torch.tanh(e)).squeeze(dim=-1)
            e.masked_fill_(src_mask == 0, -float('inf'))
            a = F.softmax(e, dim=1)
            tmp_context = torch.bmm(a.unsqueeze(1), memory)
            context.append(tmp_context)
        context = torch.cat(context, dim=1)
        feats = torch.cat([dec_out, context], dim=-1)
        feats = self.affine(feats)

        # out = self.generator(dec_out)
        out = F.log_softmax(self.generator(feats), dim=-1)

        return out.squeeze(dim=1), hidden_states

    def decode_beam(self, hidden_states, memory, src_mask, vocab,
            beam_size=5, n_best=1, alpha=0.6, length_pen='avg'):
        """
        Decode with beam search
        """
        results = {"scores":[], "predictions":[]}

        # Construct beams, we donot use stepwise coverage penalty nor ngrams block
        remaining_sents = memory.size(0)
        global_scorer = GNMTGlobalScorer(alpha, length_pen)
        beam = [ Beam(beam_size, vocab, global_scorer=global_scorer, device=memory.device)
                for _ in range(remaining_sents) ]

        # repeat beam_size times
        memory, src_mask = tile([memory, src_mask], beam_size, dim=0)
        hidden_states = tile(hidden_states, beam_size, dim=1)
        h_c = type(hidden_states) in [list, tuple]
        batch_idx = list(range(remaining_sents))

        for _ in range(MAX_DECODE_LENGTH):
            # (a) construct beamsize * remaining_sents next words
            ys = torch.stack([b.get_current_state() for b in beam if not b.done()]).contiguous().view(-1,1)

            # (b) pass through the decoder network
            out, hidden_states = self.decode_one_step(ys, hidden_states, memory, src_mask)
            out = out.contiguous().view(remaining_sents, beam_size, -1)

            # (c) advance each beam
            active, select_indices_array = [], []
            # Loop over the remaining_batch number of beam
            for b in range(remaining_sents):
                idx = batch_idx[b] # idx represent the original order in minibatch_size
                beam[idx].advance(out[b])
                if not beam[idx].done():
                    active.append((idx, b))
                select_indices_array.append(beam[idx].get_current_origin() + b * beam_size)

            # (d) update hidden_states history
            select_indices_array = torch.cat(select_indices_array, dim=0)
            if h_c:
                hidden_states = (hidden_states[0].index_select(1, select_indices_array), hidden_states[1].index_select(1, select_indices_array))
            else:
                hidden_states = hidden_states.index_select(1, select_indices_array)
            
            if not active:
                break

            # (e) reserve un-finished batches
            active_idx = torch.tensor([item[1] for item in active], dtype=torch.long, device=memory.device) # original order in remaining batch
            batch_idx = { idx: item[0] for idx, item in enumerate(active) } # order for next remaining batch

            def update_active(t):
                if t is None: return t
                t_reshape = t.contiguous().view(remaining_sents, beam_size, -1)
                new_size = list(t.size())
                new_size[0] = -1
                return t_reshape.index_select(0, active_idx).view(*new_size)

            if h_c:
                hidden_states = (
                    update_active(hidden_states[0].transpose(0, 1)).transpose(0, 1).contiguous(),
                    update_active(hidden_states[1].transpose(0, 1)).transpose(0, 1).contiguous()
                )
            else:
                hidden_states = update_active(hidden_states.transpose(0, 1)).transpose(0, 1).contiguous()
            memory = update_active(memory)
            src_mask = update_active(src_mask)
            copy_tokens = update_active(copy_tokens)
            remaining_sents = len(active)

        for b in beam:
            scores, ks = b.sort_finished(minimum=n_best)
            hyps = []
            for times, k in ks[:n_best]:
                hyp = b.get_hyp(times, k)
                hyps.append(hyp.tolist()) # hyp contains </s> but does not contain <s>
            results["predictions"].append(hyps) # batch list of variable_tgt_len
            results["scores"].append(torch.stack(scores)[:n_best]) # list of [n_best], torch.FloatTensor
        results["scores"] = torch.stack(results["scores"])

        return results

    def pad_embedding_grad_zero(self):
        self.src_embed.pad_embedding_grad_zero()
        self.tgt_embed.pad_embedding_grad_zero()

    def load_model(self, load_dir):
        self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))