"""Utility functions for loading and processing data."""

import os
import random
import json
import math

from data_utils.batch import TurnItem, TurnBatch, InteractionItem, InteractionBatch
from data_utils.split import DatasetSplit
from data_utils.interaction import load_function
from data_utils.data_vocab import DataVocab


class Corpus():
    """Contains the Text-to-SQL data.
    
    Attributes:
        train_data (DatasetSplit)
        valid_data (DatasetSplit)
        input_vocab (DataVocab)
        output_vocab (DataVocab)
        output_vocab_shema (DataVocab)
    """

    def __init__(self, params):
        if not os.path.exists(params.data_dir):
            os.mkdir(params.data_dir)

        database_schema = None
        if params.database_schema_filename:
            if 'removefrom' not in params.data_dir:
                database_schema, column_names_surface_form, column_names_embedder_input = \
                    self.read_database_schema_simple(params.database_schema_filename)
            else:
                database_schema, column_names_surface_form, column_names_embedder_input = \
                    self.read_database_schema(params.database_schema_filename)

        # interaction load function
        int_load_function = load_function(params.data_dir, database_schema)

        def collapse_list(the_list):
            """Collapses a list of list into a single list."""
            return [s for i in the_list for s in i]

        self.train_data = DatasetSplit(
            os.path.join(params.data_dir, params.processed_train_filename),
            params.raw_train_filename,
            int_load_function)
        self.valid_data = DatasetSplit(
            os.path.join(params.data_dir, params.processed_valid_filename),
            params.raw_validation_filename,
            int_load_function)

        train_input_seqs = collapse_list(self.train_data.get_ex_properties(lambda i: i.input_seqs()))
        valid_input_seqs = collapse_list(self.valid_data.get_ex_properties(lambda i: i.input_seqs()))
        all_input_seqs = train_input_seqs + valid_input_seqs

        self.input_vocab = DataVocab(
            all_input_seqs,
            os.path.join(params.data_dir, params.input_vocab_filename),
            params,
            data_type='input')

        self.output_vocab_schema = DataVocab(
            column_names_embedder_input,
            os.path.join(params.data_dir, 'schema_'+params.output_vocab_filename),
            params,
            data_type='schema')

        train_output_seqs = collapse_list(self.train_data.get_ex_properties(lambda i: i.output_seqs()))
        valid_output_seqs = collapse_list(self.valid_data.get_ex_properties(lambda i: i.output_seqs()))
        all_output_seqs = train_output_seqs + valid_output_seqs

        sql_keywords = ['.', 't1', 't2', '=', 'select', 'as', 'join', 'on', ')', '(', \
            'where', 't3', 'by', ',', 'group', 'distinct', 't4', 'and', 'limit', 'desc', \
            '>', 'avg', 'having', 'max', 'in', '<', 'sum', 't5', 'intersect', 'not', \
            'min', 'except', 'or', 'asc', 'like', '!', 'union', 'between', 't6', '-', \
            't7', '+', '/', 'count', 'from', 'value', 'order', \
            'group_by', 'order_by', 'limit_value', '!=']

        # skip column_names_surface_form and keep sql_keywords
        skip_tokens = list(set(column_names_surface_form) - set(sql_keywords))

        if params.data_dir == 'processed_data_sparc_removefrom_test': # TODO: what is this for
            all_output_seqs = []
            out_vocab_ordered = ['select', 'value', ')', '(', 'where', '=', ',', 'count', \
                'group_by', 'order_by', 'limit_value', 'desc', '>', 'distinct', 'avg', \
                'and', 'having', '<', 'in', 'max', 'sum', 'asc', 'like', 'not', 'or', \
                'min', 'intersect', 'except', '!=', 'union', 'between', '-', '+']
            for i in range(len(out_vocab_ordered)):
                all_output_seqs.append(out_vocab_ordered[:i+1])

        self.output_vocab = DataVocab(
            all_output_seqs,
            os.path.join(params.data_dir, params.output_vocab_filename),
            params,
            data_type='output',
            skip=skip_tokens)

    def read_database_schema_simple(self, database_schema_filename):
        """Reads schema for original dataset."""

        with open(database_schema_filename, 'r') as f:
            database_schema = json.load(f)

        database_schema_dict = {}
        for table_schema in database_schema:
            db_id = table_schema['db_id']
            database_schema_dict[db_id] = table_schema

            column_names = table_schema['column_names']
            column_names_original = table_schema['column_names_original']
            table_names = table_schema['table_names']
            table_names_original = table_schema['table_names_original']

            column_names_surface_form = [column_name.lower() for _, column_name in column_names_original]
            column_names_surface_form += [table_name.lower() for table_name in table_names_original]

            column_names_embedder_input = [column_name.split() for _, column_name in column_names]
            column_names_embedder_input += [table_name.split() for _, table_name in table_names]

        return database_schema_dict, column_names_surface_form, column_names_embedder_input

    def read_database_schema(self, database_schema_filename):
        """Reads schema for preprocessed dataset."""

        with open(database_schema_filename, 'r') as f:
            database_schema = json.load(f)

        database_schema_dict = {}
        column_names_surface_form = []
        column_names_embedder_input = []
        for table_schema in database_schema:
            db_id = table_schema['db_id']
            database_schema_dict[db_id] = table_schema

            column_names = table_schema['column_names']
            column_names_original = table_schema['column_names_original']
            table_names = table_schema['table_names']
            table_names_original = table_schema['table_names_original']

            for table_id, column_name in enumerate(column_names_original):
                if table_id >= 0:
                    table_name = table_names_original[table_id]
                    column_name_surface_form = '{}.{}'.format(table_name,column_name)
                else:
                    column_name_surface_form = column_name
                column_names_surface_form.append(column_name_surface_form.lower())

            # also add table_name.*
            for table_name in table_names_original:
                column_names_surface_form.append('{}.*'.format(table_name.lower()))

            for table_id, column_name in column_names:
                if table_id >= 0:
                    table_name = table_names[table_id]
                    column_name_embedder_input = table_name + ' . ' + column_name
                else:
                    column_name_embedder_input = column_name
                column_names_embedder_input.append(column_name_embedder_input.split())

            for table_name in table_names:
                column_name_embedder_input = table_name + ' . *'
                column_names_embedder_input.append(column_name_embedder_input.split())

        return database_schema_dict, column_names_surface_form, column_names_embedder_input

    def get_all_turns(
            self,
            dataset,
            max_input_length=math.inf,
            max_output_length=math.inf):
        """Returns all turns in a dataset."""

        items = []
        for interaction in dataset.examples:
            for i, turn in enumerate(interaction.turns):
                if turn.length_valid(max_input_length, max_output_length):
                    items.append(TurnItem(interaction, i))
        return items

    def get_all_interactions(
            self,
            dataset,
            max_interaction_length=math.inf,
            max_input_length=math.inf,
            max_output_length=math.inf,
            sorted_by_length=False):
        """Gets all interactions in a dataset that fit the criteria.

        Args:
            dataset (DatasetSplit): The dataset to use.
            max_interaction_length (int): Maximum interaction length to keep.
            max_input_length (int): Maximum input sequence length to keep.
            max_output_length (int): Maximum output sequence length to keep.
            sorted_by_length (bool): Whether to sort the examples by interaction length.
        """
        ints = [
            InteractionItem(
                interaction,
                max_input_length,
                max_output_length,
                max_interaction_length) for interaction in dataset.examples]
        if sorted_by_length:
            return sorted(ints, key=len)[::-1]
        else:
            return ints

    # This defines a standard way of training: each example is an turn, and
    # the batch can contain unrelated turns.
    def get_turn_batches(
            self,
            batch_size,
            max_input_length=math.inf,
            max_output_length=math.inf,
            randomize=True):
        """Gets batches of turns in the data.

        Args:
            batch_size (int): Batch size to use.
            max_input_length (int): Maximum length of input to keep.
            max_output_length (int): Maximum length of output to use.
            randomize (bool): Whether to randomize the ordering.
        """
        # First, get all interactions and the positions of the turns that are
        # possible in them.
        items = self.get_all_turns(
            self.train_data,
            max_input_length,
            max_output_length)
        if randomize:
            random.shuffle(items)

        batches = []

        current_batch_items = []
        for item in items:
            if len(current_batch_items) >= batch_size:
                batches.append(TurnBatch(current_batch_items))
                current_batch_items = []
            current_batch_items.append(item)
        batches.append(TurnBatch(current_batch_items))

        assert sum([len(batch) for batch in batches]) == len(items)

        return batches

    def get_interaction_batches(
            self,
            batch_size,
            max_interaction_length=math.inf,
            max_input_length=math.inf,
            max_output_length=math.inf,
            randomize=True):
        """Gets batches of interactions in the data.

        Args:
            batch_size (int): Batch size to use.
            max_interaction_length (int): Maximum length of interaction to keep
            max_input_length (int): Maximum length of input to keep.
            max_output_length (int): Maximum length of output to keep.
            randomize (bool): Whether to randomize the ordering.
        """
        items = self.get_all_interactions(
            self.train_data,
            max_interaction_length,
            max_input_length,
            max_output_length,
            sorted_by_length=not randomize)
        if randomize:
            random.shuffle(items)

        batches = []
        current_batch_items = []
        for item in items:
            if len(current_batch_items) >= batch_size:
                batches.append(
                    InteractionBatch(current_batch_items))
                current_batch_items = []
            current_batch_items.append(item)
        batches.append(InteractionBatch(current_batch_items))

        assert sum([len(batch) for batch in batches]) == len(items)

        return batches

    def get_random_turns(
            self,
            num_samples,
            max_input_length=math.inf,
            max_output_length=math.inf):
        """Gets a random selection of turns in the data.

        Args:
            num_samples (bool): Number of random turns to get.
            max_input_length (int): Limit of input length.
            max_output_length (int): Limit on output length.
        """
        items = self.get_all_turns(
            self.train_data,
            max_input_length,
            max_output_length)
        random.shuffle(items)
        return items[:num_samples]

    def get_random_interactions(
            self,
            num_samples,
            max_interaction_length=math.inf,
            max_input_length=math.inf,
            max_output_length=math.inf):
        """Gets a random selection of interactions in the data.

        Args:
            num_samples (bool): Number of random interactions to get.
            max_input_length (int): Limit of input length.
            max_output_length (int): Limit on output length.
        """
        items = self.get_all_interactions(
            self.train_data,
            max_interaction_length,
            max_input_length,
            max_output_length)
        random.shuffle(items)
        return items[:num_samples]


def num_turns(dataset):
    """Returns the total number of turns in the dataset.
    
    Args:
        dataset (DatasetSplit)
    """
    return sum([len(interaction) for interaction in dataset.examples])
