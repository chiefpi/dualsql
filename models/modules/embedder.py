"""Embedder for tokens."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils.vocab import UNK_TOK


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
        for index, token in vocab.id2token:
            if token in glove_embs:
                id2emb[index] = glove_embs[token]
            else:
                glove_oov += 1
    
        print('Glove OOV:', glove_oov, 'Total', len(vocab))
    
        return torch.FloatTensor(id2emb)
  
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
        for index, token in vocab.id2token:
            if token in glove_embs:
                id2emb[index] = glove_embs[token]
            else:
                glove_oov += 1
    
        print('Glove OOV:', glove_oov, 'Total', len(vocab))
    
        return id2emb
  
    utter_vocab_embs = create_word_embs(utter_vocab)
    query_vocab_embs = create_word_embs(query_vocab)
    schema_vocab_embs = create_word_embs(schema_vocab) if schema_vocab else None
  
    return utter_vocab_embs, query_vocab_embs, schema_vocab_embs, glove_emb_dim


class Embedder(nn.Module):
    """Embeds tokens."""

    def __init__(
            self,
            emb_size,
            init=None,
            vocab=None,
            num_tokens=-1,
            freeze=False,
            use_unk=True):
        super().__init__()

        if vocab:
            assert num_tokens < 0, "Specified a vocab but also set number of tokens to " + \
                str(num_tokens)
            self.in_vocab = lambda token: token in vocab.tokens
            self.vocab_token_lookup = lambda token: vocab.token_to_id(token)
            if use_unk:
                self.unknown_token_id = vocab.token_to_id(UNK_TOK)
            else:
                self.unknown_token_id = -1
            self.vocab_size = len(vocab)
        else:
            def check_vocab(index):
                """Makes sure the index is in the vocab."""
                assert index < num_tokens, "Passed token ID " + \
                    str(index) + "; expecting something less than " + str(num_tokens)
                return index < num_tokens
            self.in_vocab = check_vocab
            self.vocab_token_lookup = lambda x: x
            self.unknown_token_id = num_tokens  # Deliberately throws an error here,
            # But should crash before this
            self.vocab_size = num_tokens

        if init is not None:
            word_embeddings_tensor = torch.FloatTensor(init)
            self.token_embedding_matrix = torch.nn.Embedding.from_pretrained(
                word_embeddings_tensor, freeze=freeze)
        else:
            init_tensor = torch.empty(self.vocab_size, emb_size).uniform_(-0.1, 0.1)
            self.token_embedding_matrix = torch.nn.Embedding.from_pretrained(
                init_tensor, freeze=False)

    def forward(self, tokens):
        """
        Args:
            tokens (list of int)

        Returns:
            len x emb_dim
        """

        index_list = torch.LongTensor([self.vocab_token_lookup(token) 
            if self.in_vocab(token) else self.unknown_token_id
            for token in tokens])

        if self.token_embedding_matrix.weight.is_cuda:
            index_list = index_list.cuda()
        return self.token_embedding_matrix(index_list).squeeze()
