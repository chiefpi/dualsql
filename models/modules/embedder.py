"""Embedder for tokens."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_vocab_embs(vocab, emb_file):
    """
    Returns:
        torch.FloatTensor: Glove embeddings
    """
    def read_glove_emb(emb_file, emb_dim):
        glove_embs = {}
    
        with open(emb_file) as f:
            for line in f:
                l_split = line.split()
                word = ' '.join(l_split[:-emb_dim])
                emb = np.array([float(val) for val in l_split[-emb_dim:]])
                glove_embs[word] = emb
    
        return glove_embs
  
    print('Loading Glove Embedding from', emb_file)
    glove_emb_dim = 300
    glove_embs = read_glove_emb(emb_file, glove_emb_dim)
    print('Done')

    def create_word_embs(vocab):
        id2emb = np.zeros((len(vocab), glove_emb_dim), dtype=np.float32)
    
        glove_oov = 0
        for index, token in enumerate(vocab.id2token):
            if token in glove_embs:
                id2emb[index] = glove_embs[token]
            else:
                glove_oov += 1
    
        print('Glove OOV:', glove_oov, 'Total', len(vocab))
    
        return torch.FloatTensor(id2emb).to(device)
  
    vocab_embs = create_word_embs(vocab)
  
    return vocab_embs, glove_emb_dim


def load_all_embs(utter_vocab, query_vocab, schema_vocab, emb_file):
    print(query_vocab.id2token)
    print()
  
    def read_glove_emb(emb_file, emb_dim):
        glove_embs = {}
    
        with open(emb_file) as f:
            for line in f:
                l_split = line.split()
                word = ' '.join(l_split[:-emb_dim])
                emb = np.array([float(val) for val in l_split[-emb_dim:]])
                glove_embs[word] = emb
    
        return glove_embs
  
    print('Loading Glove Embedding from', emb_file)
    glove_emb_dim = 300
    glove_embs = read_glove_emb(emb_file, glove_emb_dim)
    print('Done')

    def create_word_embs(vocab):
        id2emb = np.zeros((len(vocab), glove_emb_dim), dtype=np.float32)
    
        glove_oov = 0
        for index, token in enumerate(vocab.id2token):
            if token in glove_embs:
                id2emb[index] = glove_embs[token]
            else:
                glove_oov += 1
    
        print('Glove OOV:', glove_oov, 'Total', len(vocab))
    
        return torch.FloatTensor(id2emb).to(device)
  
    utter_vocab_embs = create_word_embs(utter_vocab)
    query_vocab_embs = create_word_embs(query_vocab)
    schema_vocab_embs = create_word_embs(schema_vocab) if schema_vocab else None
  
    return utter_vocab_embs, query_vocab_embs, schema_vocab_embs, glove_emb_dim


class Embedder(nn.Module):
    """Contains an embedding matrix and a vocabulary."""

    def __init__(self, emb_size, vocab, init=None, freeze=False):
        super(Embedder, self).__init__()

        self.vocab = vocab

        if init is None:
            self.emb = nn.Embedding(len(vocab), emb_size)
        else:
            self.emb = nn.Embedding.from_pretrained(init, freeze=freeze)

    def forward(self, ids):
        """
        Args:
            ids (list of int)

        Returns:
            torch.FloatTensor (len x emb_dim)
        """

        return self.emb(torch.LongTensor(ids).to(device))
