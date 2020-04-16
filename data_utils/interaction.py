"""Contains the class for an interaction."""
import torch

from data_utils.turn import Turn


class Schema:
    # TODO: rewrite this
    def __init__(self, table_schema, simple=False):
        if simple:
            self.init_simple(table_schema)
        else:
            self.init(table_schema)

    def init_simple(self, table_schema):
        self.table_schema = table_schema
        column_names = table_schema['column_names']
        column_names_original = table_schema['column_names_original']
        table_names = table_schema['table_names']
        table_names_original = table_schema['table_names_original']
        assert len(column_names) == len(column_names_original) and len(table_names) == len(table_names_original)

        column_keep_index = []

        self.col_names_surface = []
        self.col_names_to_id = {}
        for i, (table_id, col_name) in enumerate(column_names_original):
            col_name = col_name.lower()
            if col_name not in self.col_names_to_id:
                self.col_names_surface.append(col_name)
                self.col_names_to_id[col_name] = len(self.col_names_surface) - 1
                column_keep_index.append(i)

        column_keep_index_2 = []
        for i, table_name in enumerate(table_names_original):
            col_name = table_name.lower()
            if col_name not in self.col_names_to_id:
                self.col_names_surface.append(col_name)
                self.col_names_to_id[col_name] = len(self.col_names_surface) - 1
                column_keep_index_2.append(i)

        self.col_names_emb_input = []
        self.col_names_emb_input_to_id = {}
        for i, (table_id, column_name) in enumerate(column_names):
            column_name_embedder_input = column_name
            if i in column_keep_index:
                self.col_names_emb_input.append(column_name_embedder_input)
                self.col_names_emb_input_to_id[column_name_embedder_input] = len(self.col_names_emb_input) - 1

        for i, table_name in enumerate(table_names):
            column_name_embedder_input = table_name
            if i in column_keep_index_2:
                self.col_names_emb_input.append(column_name_embedder_input)
                self.col_names_emb_input_to_id[column_name_embedder_input] = len(self.col_names_emb_input) - 1

        max_id_1 = max(v for k,v in self.col_names_to_id.items())
        max_id_2 = max(v for k,v in self.col_names_emb_input_to_id.items())
        assert (len(self.col_names_surface) - 1) == max_id_2 == max_id_1

        self.num_col = len(self.col_names_surface)

    def init(self, table_schema):
        self.table_schema = table_schema
        column_names = table_schema['column_names']
        column_names_original = table_schema['column_names_original']
        table_names = table_schema['table_names']
        table_names_original = table_schema['table_names_original']
        assert len(column_names) == len(column_names_original) and len(table_names) == len(table_names_original)

        column_keep_index = []

        self.col_names_surface = []
        self.col_names_to_id = {}
        for i, (table_id, column_name) in enumerate(column_names_original):
            if table_id >= 0:
                table_name = table_names_original[table_id]
                col_name = '{}.{}'.format(table_name,column_name)
            else:
                col_name = column_name
            col_name = col_name.lower()
            if col_name not in self.col_names_to_id:
                self.col_names_surface.append(col_name)
                self.col_names_to_id[col_name] = len(self.col_names_surface) - 1
                column_keep_index.append(i)

        start_i = len(self.col_names_to_id)
        for i, table_name in enumerate(table_names_original):
            col_name = '{}.*'.format(table_name.lower())
            self.col_names_surface.append(col_name)
            self.col_names_to_id[col_name] = i + start_i

        self.col_names_emb_input = []
        self.col_names_emb_input_to_id = {}
        for i, (table_id, column_name) in enumerate(column_names):
            if table_id >= 0:
                table_name = table_names[table_id]
                column_name_embedder_input = table_name + ' . ' + column_name
            else:
                column_name_embedder_input = column_name
            if i in column_keep_index:
                self.col_names_emb_input.append(column_name_embedder_input)
                self.col_names_emb_input_to_id[column_name_embedder_input] = len(self.col_names_emb_input) - 1

        start_i = len(self.col_names_emb_input_to_id)
        for i, table_name in enumerate(table_names):
            column_name_embedder_input = table_name + ' . *'
            self.col_names_emb_input.append(column_name_embedder_input)
            self.col_names_emb_input_to_id[column_name_embedder_input] = i + start_i

        assert len(self.col_names_surface) == len(self.col_names_to_id) == len(self.col_names_emb_input) == len(self.col_names_emb_input_to_id)

        assert len(self.col_names_surface)-1 == \
            max(v for k,v in self.col_names_to_id.items()) == \
            max(v for k,v in self.col_names_emb_input_to_id.items())

        self.num_col = len(self.col_names_surface)

    def __len__(self):
        return self.num_col

    def in_vocabulary(self, column_name, surface_form=False):
        if surface_form:
            return column_name in self.col_names_to_id
        else:
            return column_name in self.col_names_emb_input_to_id

    def column_name_embedder_bow(self, column_name, surface_form=False, column_name_token_embedder=None):
        assert self.in_vocabulary(column_name, surface_form)
        if surface_form:
            column_name_id = self.col_names_to_id[column_name]
            column_name_embedder_input = self.col_names_emb_input[column_name_id]
        else:
            column_name_embedder_input = column_name

        column_name_embeddings = [column_name_token_embedder(token) for token in column_name_embedder_input.split()]
        column_name_embeddings = torch.stack(column_name_embeddings, dim=0)
        return torch.mean(column_name_embeddings, dim=0)

    def set_column_name_embeddings(self, column_name_embeddings):
        self.column_name_embeddings = column_name_embeddings
        assert len(self.column_name_embeddings) == self.num_col

    def column_name_embedder(self, column_name, surface_form=False):
        assert self.in_vocabulary(column_name, surface_form)
        if surface_form:
            column_name_id = self.col_names_to_id[column_name]
        else:
            column_name_id = self.col_names_emb_input_to_id[column_name]

        return self.column_name_embeddings[column_name_id]


class Interaction:
    """Contains an interaction.

    Attributes:
        turns (list of Turn): Turns in an interaction.
        schema (Schema)
        identifier (str): Identifier for an interaction.
    """
    def __init__(self, turns, schema, identifier):
        self.turns = turns
        self.schema = schema
        self.identifier = identifier

    def __str__(self):
        return '\n'.join(['Turns:'] + [str(turn) for turn in self.turns])

    def __len__(self):
        return len(self.turns)

    def input_seqs(self):
        return [turn.input_seq for turn in self.turns]

    def output_seqs(self):
        return [turn.output_seq for turn in self.turns]


def load_function(data_dir, db_schema=None):
    def fn(interaction_example):
        keep = False

        raw_turns = interaction_example['interaction']

        if 'database_id' in interaction_example:
            database_id = interaction_example['database_id']
            interaction_id = interaction_example['interaction_id']
            identifier = str(database_id) + '/' + str(interaction_id)
        else:
            identifier = interaction_example['id']

        schema = None
        if db_schema:
            if 'removefrom' not in data_dir:
                schema = Schema(db_schema[database_id], simple=True)
            else:
                schema = Schema(db_schema[database_id])

        turn_examples = []

        for turn in raw_turns:
            proc_turn = Turn(turn)
            keep_turns = proc_turn.keep
            assert not schema or keep_turns
            if keep_turns:
                keep = True
                turn_examples.append(proc_turn)

        interaction = Interaction(
            turn_examples,
            schema,
            identifier)

        return interaction, keep

    return fn
