"""Contains the Turn class and token functions."""

import nltk
import sqlparse

from data_utils.vocab import BOS_TOK, EOS_TOK, UNK_TOK


class Turn:
    """Contains a turn in an interaction.
    
    Attributes:
        utter_seq (list of str): nl tokens.
        query_seq (list of str): sql tokens.
        keep (bool): if not empty.
    """

    def __init__(self, example):
        self.utter_seq = [BOS_TOK] + nltk.word_tokenize(example['utterance']) + [EOS_TOK]
        self.query_seq = [BOS_TOK] + example['sql'] + [EOS_TOK]
        self.keep = self.query_seq and self.utter_seq

    def __str__(self):
        return 'Utter: ' + ' '.join(self.utter_seq) + '\n' + \
            'Query: ' + ' '.join(self.query_seq) + '\n'

    def length_valid(self, utter_limit, query_limit):
        return len(self.utter_seq) < utter_limit \
            and len(self.query_seq) < query_limit

    def str2index(self, schema, utter_vocab, query_vocab):
        self.utter_seq_id = [utter_vocab.token2id[t] for t in self.utter_seq]
        offset = len(query_vocab)
        self.query_seq_id = [query_vocab.token2id[t] if t in query_vocab.token2id
            else schema.vocab.token2id[t]+offset for t in self.query_seq] # column names

    # def index2str(self, schema, utter_vocab, query_vocab):
    #     self.utter_seq = [utter_vocab.id2token[i] for i in self.utter_seq_id]
    #     offset = len(query_vocab)
    #     self.query_seq = [query_vocab.id2token[i] if i < offset
    #         else schema.vocab.id2token[i-offset] for i in self.query_seq_id] # column names
