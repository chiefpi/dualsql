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

    def str2index(self, utter_vocab, query_vocab):
        self.utter_seq = [utter_vocab.token2id[t] for t in self.utter_seq]
        self.query_seq = [query_vocab.token2id[t] if t in query_vocab.token2id
            else query_vocab.token2id[UNK_TOK] for t in self.query_seq]