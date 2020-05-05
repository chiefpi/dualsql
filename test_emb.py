import torch
from models.modules.embedder import Embedder

embedder = Embedder(100, num_tokens=300)

tokens = [0, 1, 2, 3, 4]
print(embedder(tokens).size())