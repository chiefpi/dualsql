"""Utility functions for loading and processing data."""

import os
import random
import json
import math

import torch

from data_utils.split import DatasetSplit
from data_utils.vocab import Vocab
from data_utils.schema import load_db_schema


class Corpus:
    """Contains SParC dataset.
    
    Attributes:
        train_data (DatasetSplit)
        valid_data (DatasetSplit)
        schema_vocab (Vocab)
        utter_vocab (Vocab)
        query_vocab (Vocab)
    """

    def __init__(self, params):
        if not os.path.exists(params.data_dir):
            os.mkdir(params.data_dir)

        db2schema, all_schema_tokens, all_schema_tokens_sep = load_db_schema(
            os.path.join(params.raw_data_dir, params.db_schema_filename),
            params.remove_from)

        self.train_data = DatasetSplit(
            os.path.join(params.data_dir, 'train.pkl'),
            os.path.join(params.raw_data_dir, 'train.pkl'),
            db2schema)

        self.valid_data = DatasetSplit(
            os.path.join(params.data_dir, 'dev.pkl'),
            os.path.join(params.raw_data_dir, 'dev.pkl'),
            db2schema)

        all_utter_seqs = self.train_data.get_all_utterances() + self.valid_data.get_all_utterances()
        # all_query_seqs = self.train_data.get_all_queries() + self.valid_data.get_all_queries()

        sql_keywords = ['select', ')', '(', 'value', 'count', 'where', ',', '=', 'group_by', 'order_by', 'limit_value', 'desc', 'distinct', '>', 'avg', 'having', 'and', '<', 'asc', 'in', 'sum', 'max', 'except', 'not', 'intersect', 'or', 'min', 'like', '!=', 'union', 'between', '-', '+']

        # Build vocabularies
        self.schema_vocab = Vocab(all_schema_tokens_sep, data_type='schema')
        self.utter_vocab = Vocab(all_utter_seqs, data_type='utter')
        # skip_tokens = list(set(all_schema_tokens) - set(sql_keywords)) # skip column names
        # self.query_vocab = Vocab(all_query_seqs, data_type='query', skip=skip_tokens)
        self.query_vocab = Vocab([sql_keywords], data_type='query')

        self.train_data.str2index(self.schema_vocab, self.utter_vocab, self.query_vocab)
        self.valid_data.str2index(self.schema_vocab, self.utter_vocab, self.query_vocab)

        # self.train_data.index2str(self.schema_vocab, self.utter_vocab, self.query_vocab)
        # self.valid_data.index2str(self.schema_vocab, self.utter_vocab, self.query_vocab)
