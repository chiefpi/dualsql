import torch
from models.modules.embedder import Embedder
from data_utils.vocab import Vocab

embedder = Embedder(5, Vocab([['just', 'a', 'test']], data_type='utter'), freeze=True)

tokens = [0, 1, 2, 3, 4, 5, 6]
print(embedder(tokens))