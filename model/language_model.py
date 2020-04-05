import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tensors import lens2mask


class LanguageModel(nn.Module):
    """LSTM Language Model
    """
    def __init__(
            self, vocab_size, emb_dim, hidden_dim, num_layers,
            pad_token_idxs=[], dropout=0.5, tie_weights=False):
        super(LanguageModel, self).__init__()

        # self.pad_token_idxs = list(pad_token_idxs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)
        self.embedder = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(
            emb_dim, hidden_dim, num_layers,
            dropout=(dropout if num_layers > 1 else 0))
        self.decoder = nn.Linear(hidden_dim, vocab_size)

        if tie_weights:
            if hidden_dim != emb_dim:
                raise ValueError('When using the tied flag, hidden_dim must be equal to emb_dim')
            self.decoder.weight = self.embedder.weight

        self.init_weights()

    def init_weights(self, init_range=0.1):
        # for pad_token_idx in self.pad_token_idxs:
        #     self.embedder.weight.data[pad_token_idx].zero_()
        self.embedder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_dim),
            weight.new_zeros(self.num_layers, batch_size, self.hidden_dim))

    # def pad_embedding_grad_zero(self):
    #     for pad_token_idx in self.pad_token_idxs:
    #         self.embedder.weight.grad[pad_token_idx].zero_()

    def forward(self, input, hidden):
        emb = self.dropout(self.embedder(input)) # bsize, seq_length, emb_dim
        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder(self.dropout(output))
        return F.log_softmax(decoded, dim=-1)

    def sentence_log_prob(self, input, hidden):
        """Calculate length-normalized log-probability of a sentence.
        Sequence must contain BOS and EOS symbol. TODO: wtf
        :param input: sentence
        :param hidden: hidden state
        :return: length-normalized log-probability
        """
        # output = input[1:] # EOS?
        score = self.forward(input[:-1], hidden)

        # log_prob = torch.gather(scores, 2, output.unsqueeze(-1)).contiguous().view(output.size(0), output.size(1))
        # sent_log_prob = torch.sum(log_prob * lens2mask(lens).float(), dim=-1)
        return score / len(input)

    def load_model(self, load_dir):
        self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))