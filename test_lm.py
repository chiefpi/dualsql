import torch
from models.language_model import LanguageModel

lm = LanguageModel(
    vocab_size=1000,
    emb_dim=100,
    hidden_dim=100,
    num_layers=2,
    dropout=0.5,
    tie_weights=True)

sentences = torch.LongTensor([[2,21,34,999],[0,1,2,3]])
print(sentences.size())
lens = torch.LongTensor([4,2])
print(lm.sentence_log_prob(sentences, lens))