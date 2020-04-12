import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils.vocabulary import BOS_TOK, EOS_TOK

from modules.embedder import Embedder, load_vocab_embs


class LanguageModel(nn.Module):
    """LSTM Language Model"""
    def __init__(
            self, vocab, emb_file, hidden_dim=300, num_layers=2,
            dropout=0.5, tie_weights=False, freeze=True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)

        vocab_size = len(vocab)
        vocab_embs, emb_dim = load_vocab_embs(vocab, emb_file)
        self.embedder = Embedder(
            emb_dim,
            initializer=vocab_embs,
            vocabulary=vocab,
            freeze=freeze)

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
        self.embedder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_dim),
            weight.new_zeros(self.num_layers, batch_size, self.hidden_dim))

    def forward(self, input, hidden):
        emb = self.dropout(self.embedder(input)) # bsize x seq_len x emb_dim
        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder(self.dropout(output))
        return F.log_softmax(decoded, dim=-1)

    def sentence_log_prob(self, input, hidden):
        """Calculates length-normalized log-probability of a batch.

        Args:
            input: Sentence must contain BOS and EOS symbol.
            hidden: Hidden states

        Returns:
            float: Length-normalized log-probability
        """
        # TODO: batch
        # output = input[1:] # EOS?
        score = self.forward(input[:-1], hidden)

        # log_prob = torch.gather(scores, 2, output.unsqueeze(-1)).contiguous().view(output.size(0), output.size(1))
        # sent_log_prob = torch.sum(log_prob * lens2mask(lens).float(), dim=-1)
        return score / len(input)

    def load_model(self, load_dir):
        self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))