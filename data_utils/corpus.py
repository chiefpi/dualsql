"""Utility functions for loading and processing data."""

import os
import random
import json
import math

from data_utils.split import DatasetSplit
from data_utils.vocab import Vocab
from data_utils.schema import load_db_schema


class Corpus:
    """Contains the Text-to-SQL data.
    
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
            os.path.join(params.data_dir, 'valid.pkl'),
            os.path.join(params.raw_data_dir, 'valid.pkl'),
            db2schema)

        all_utter_seqs = self.train_data.get_all_utterances() + self.valid_data.get_all_utterances()
        all_query_seqs = self.train_data.get_all_queries() + self.valid_data.get_all_queries()

        sql_keywords = ['.', 't1', 't2', '=', 'select', 'as', 'join', 'on', ')', '(', \
            'where', 't3', 'by', ',', 'group', 'distinct', 't4', 'and', 'limit', 'desc', \
            '>', 'avg', 'having', 'max', 'in', '<', 'sum', 't5', 'intersect', 'not', \
            'min', 'except', 'or', 'asc', 'like', '!', 'union', 'between', 't6', '-', \
            't7', '+', '/', 'count', 'from', 'value', 'order', \
            'group_by', 'order_by', 'limit_value', '!=']

        # Build vocabularies
        self.schema_vocab = Vocab(all_schema_tokens_sep, data_type='schema')
        self.utter_vocab = Vocab(all_utter_seqs, data_type='utter')
        # Skip non-keywords
        skip_tokens = list(set(all_schema_tokens) - set(sql_keywords))
        self.query_vocab = Vocab(all_query_seqs, data_type='query', skip=skip_tokens)

        self.train_data.str2index()
        self.valid_data.str2index()
        # if params.data_dir == 'processed_data_sparc_removefrom_test': # TODO: what is this for
        #     all_query_seqs = []
        #     out_vocab_ordered = ['select', 'value', ')', '(', 'where', '=', ',', 'count', \
        #         'group_by', 'order_by', 'limit_value', 'desc', '>', 'distinct', 'avg', \
        #         'and', 'having', '<', 'in', 'max', 'sum', 'asc', 'like', 'not', 'or', \
        #         'min', 'intersect', 'except', '!=', 'union', 'between', '-', '+']
        #     for i in range(len(out_vocab_ordered)):
        #         all_query_seqs.append(out_vocab_ordered[:i+1])

    def get_all_turns(
            self,
            dataset,
            max_utter_len=math.inf,
            max_query_len=math.inf):
        """Gets all turns in a dataset.
        
        Returns:
            list of Turn
        """

        return [turn for interaction in dataset.interactions
            for turn in interaction.turns
            if turn.len_valid(max_utter_len, max_query_len)]

    def get_all_interactions(
            self,
            dataset,
            max_inter_len=math.inf,
            max_utter_len=math.inf,
            max_query_len=math.inf,
            sorted_by_len=False):
        """Gets all interactions in a dataset that fit the criteria.

        Args:
            dataset (DatasetSplit): The dataset to use.
            max_inter_len (int): Maximum interaction len to keep.
            max_utter_len (int): Maximum utter sequence len to keep.
            max_query_len (int): Maximum query sequence len to keep.
            sorted_by_len (bool): Whether to sort the interactions by interaction len.

        Returns:
            list of Interaction
        """
        interactions = [interaction for interaction in dataset.interactions
            if len(interaction) <= max_inter_len]
        for interaction in interactions:
            interaction.set_valid_len(max_utter_len, max_query_len) # TODO

        return sorted(interactions, key=len, reverse=True) if sorted_by_len else interactions

    def get_turn_batches( # TODO
            self,
            batch_size,
            max_utter_len=math.inf,
            max_query_len=math.inf,
            randomize=True):
        """Gets batches of turns in the data.

        Args:
            batch_size (int): Batch size to use.
            max_utter_len (int): Maximum len of utter to keep.
            max_query_len (int): Maximum len of query to use.
            randomize (bool): Whether to randomize the ordering.

        Returns:
            list of list of Turn
        """
        turns = self.get_all_turns(
            self.train_data,
            max_utter_len,
            max_query_len)
        if randomize:
            random.shuffle(turns)

        return [turns[i:i+batch_size]
            for i in range(0, len(turns), batch_size)]

    def get_interaction_items(
            self,
            max_inter_len=math.inf,
            max_utter_len=math.inf,
            max_query_len=math.inf,
            randomize=True):
        """Gets batches of interactions in the data.

        Args:
            batch_size (int): Batch size to use.
            max_inter_len (int): Maximum len of interaction to keep
            max_utter_len (int): Maximum len of utter to keep.
            max_query_len (int): Maximum len of query to keep.
            randomize (bool): Whether to randomize the ordering.
        """
        interactions = self.get_all_interactions(
            self.train_data,
            max_inter_len,
            max_utter_len,
            max_query_len,
            sorted_by_len=not randomize)
        if randomize:
            random.shuffle(interactions)

        return interactions

    def get_random_turns(
            self,
            num_samples,
            max_utter_len=math.inf,
            max_query_len=math.inf):
        """Gets a random selection of turns in the data.

        Args:
            num_samples (bool): Number of random turns to get.
            max_utter_len (int): Limit of utter len.
            max_query_len (int): Limit on query len.
        """
        items = self.get_all_turns(
            self.train_data,
            max_utter_len,
            max_query_len)
        random.shuffle(items)

        return items[:num_samples]

    def get_random_interactions(
            self,
            num_samples,
            max_inter_len=math.inf,
            max_utter_len=math.inf,
            max_query_len=math.inf):
        """Gets a random selection of interactions in the data.

        Args:
            num_samples (bool): Number of random interactions to get.
            max_utter_len (int): Limit of utter len.
            max_query_len (int): Limit on query len.
        """
        items = self.get_all_interactions(
            self.train_data,
            max_inter_len,
            max_utter_len,
            max_query_len)
        random.shuffle(items)

        return items[:num_samples]
