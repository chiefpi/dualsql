"""Embedder for tokens."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils.vocab import UNK_TOK


def load_vocab_embs(vocab, emb_file):

    def read_glove_emb(emb_file, emb_dim):
        glove_embs = {}
    
        with open(emb_file) as f:
            cnt = 1
            for line in f:
                cnt += 1
                l_split = line.split()
                word = " ".join(l_split[0:len(l_split)-emb_dim])
                emb = np.array([float(val) for val in l_split[-emb_dim:]])
                glove_embs[word] = emb
    
        return glove_embs
  
    print('Loading Glove Embedding from', emb_file)
    glove_emb_dim = 300
    glove_embs = read_glove_emb(emb_file, glove_emb_dim)
    print('Done')

    def create_word_embs(vocab):
        vocab_embs = np.zeros((len(vocab), glove_emb_dim), dtype=np.float32)
        vocab_tokens = vocab.inorder_tokens
    
        glove_oov = 0
        para_oov = 0
        for token in vocab_tokens:
            token_id = vocab.token_to_id(token)
            if token in glove_embs:
                vocab_embs[token_id][:glove_emb_dim] = glove_embs[token] # TODO: unnecessary slice
            else:
                glove_oov += 1
    
        print('Glove OOV:', glove_oov, 'Para OOV', para_oov, 'Total', len(vocab))
    
        return vocab_embs
  
    vocab_embs = create_word_embs(vocab)
  
    return vocab_embs, glove_emb_dim


def load_all_embs(input_vocab, output_vocab, output_vocab_schema, emb_file):
    print(output_vocab.inorder_tokens)
    print()
  
    def read_glove_emb(emb_file, emb_size):
        glove_embs = {}
    
        with open(emb_file) as f:
            cnt = 1
            for line in f:
                cnt += 1
                l_split = line.split()
                word = " ".join(l_split[0:len(l_split)-emb_size])
                emb = np.array([float(val) for val in l_split[-emb_size:]])
                glove_embs[word] = emb
    
        return glove_embs
  
    print('Loading Glove Embedding from', emb_file)
    glove_emb_size = 300
    glove_embs = read_glove_emb(emb_file, glove_emb_size)
    print('Done')
  
    input_emb_size = glove_emb_size
  
    def create_word_embs(vocab):
        vocab_embs = np.zeros((len(vocab), glove_emb_size), dtype=np.float32)
        vocab_tokens = vocab.inorder_tokens
    
        glove_oov = 0
        para_oov = 0
        for token in vocab_tokens:
            token_id = vocab.token_to_id(token)
            if token in glove_embs:
                vocab_embs[token_id][:glove_emb_size] = glove_embs[token]
            else:
                glove_oov += 1
    
        print('Glove OOV:', glove_oov, 'Para OOV', para_oov, 'Total', len(vocab))
    
        return vocab_embs
  
    input_vocab_embs = create_word_embs(input_vocab)
    output_vocab_embs = create_word_embs(output_vocab)
    output_vocab_schema_embs = None
    if output_vocab_schema:
        output_vocab_schema_embs = create_word_embs(output_vocab_schema)
  
    return input_vocab_embs, output_vocab_embs, output_vocab_schema_embs, input_emb_size


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

    def forward(self, token):
        assert isinstance(token, int)

        if self.in_vocab(token):
            index_list = torch.LongTensor([self.vocab_token_lookup(token)])
            if self.token_embedding_matrix.weight.is_cuda:
                index_list = index_list.cuda()
            return self.token_embedding_matrix(index_list).squeeze()
        elif self.anonymizer and self.anonymizer.is_anon_tok(token):
            index_list = torch.LongTensor([self.anonymizer.get_anon_id(token)])
            if self.token_embedding_matrix.weight.is_cuda:
                index_list = index_list.cuda()
            return self.entity_embedding_matrix(index_list).squeeze()
        else:
            index_list = torch.LongTensor([self.unknown_token_id])
            if self.token_embedding_matrix.weight.is_cuda:
                index_list = index_list.cuda()
            return self.token_embedding_matrix(index_list).squeeze()

