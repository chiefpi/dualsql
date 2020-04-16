import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils.tensor import lens2mask
from modules.embedder import Embedder, load_vocab_embs


class LanguageModel(nn.Module):
    """Language Model for utterances and queries."""
    def __init__(
            self,
            vocab_size,
            emb_dim,
            hidden_dim=300,
            num_layers=2,
            dropout=0.5,
            tie_weights=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)
        self.embedder = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
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

    def forward(self, sentences, lens):
        emb = self.dropout(self.embedder(sentences)) # bsize x max_len x emb_dim
        output, _ = self.rnn(emb) # bsize x max_len x hidden_dim
        decoded = self.decoder(self.dropout(output)) # bsize x max_len x vocab_size
        return F.log_softmax(decoded, dim=-1)

    def sentence_log_prob(self, sentences, lens):
        """Calculates length-normalized log-probability of a batch.

        Args:
            sentences: Sentences must contain EOS symbol. bsize x max_len.
            lens: Lengths of sentences.

        Returns:
            Length-normalized log-probability.
        """
        scores = self.forward(sentences[:, :-1], lens) # bsize x max_len x vocab_size

        log_prob = torch.gather(scores, 2, sentences.unsqueeze(-1)) \
            .contiguous().view(sentences.size(0), sentences.size(1))
        sentence_log_prob = torch.sum(log_prob*lens2mask(lens).float(), dim=-1)
        return sentence_log_prob / lens.float()

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        print("Loaded model from file " + filename)

    def save(self, filename):
        torch.save(self.state_dict(), filename)