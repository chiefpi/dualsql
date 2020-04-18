import torch

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

        self.column_names_surface = []
        self.column_names_to_id = {}
        for i, (table_id, col_name) in enumerate(column_names_original):
            col_name = col_name.lower()
            if col_name not in self.column_names_to_id:
                self.column_names_surface.append(col_name)
                self.column_names_to_id[col_name] = len(self.column_names_surface) - 1
                column_keep_index.append(i)

        column_keep_index_2 = []
        for i, table_name in enumerate(table_names_original):
            col_name = table_name.lower()
            if col_name not in self.column_names_to_id:
                self.column_names_surface.append(col_name)
                self.column_names_to_id[col_name] = len(self.column_names_surface) - 1
                column_keep_index_2.append(i)

        self.column_names_emb_input = []
        self.column_names_emb_input_to_id = {}
        for i, (table_id, column_name) in enumerate(column_names):
            column_name_embedder_input = column_name
            if i in column_keep_index:
                self.column_names_emb_input.append(column_name_embedder_input)
                self.column_names_emb_input_to_id[column_name_embedder_input] = len(self.column_names_emb_input) - 1

        for i, table_name in enumerate(table_names):
            column_name_embedder_input = table_name
            if i in column_keep_index_2:
                self.column_names_emb_input.append(column_name_embedder_input)
                self.column_names_emb_input_to_id[column_name_embedder_input] = len(self.column_names_emb_input) - 1

        max_id_1 = max(v for k,v in self.column_names_to_id.items())
        max_id_2 = max(v for k,v in self.column_names_emb_input_to_id.items())
        assert (len(self.column_names_surface) - 1) == max_id_2 == max_id_1

        self.num_col = len(self.column_names_surface)

    def init(self, table_schema):
        self.table_schema = table_schema
        column_names = table_schema['column_names']
        column_names_original = table_schema['column_names_original']
        table_names = table_schema['table_names']
        table_names_original = table_schema['table_names_original']
        assert len(column_names) == len(column_names_original) and len(table_names) == len(table_names_original)

        column_keep_index = []

        self.column_names_surface = []
        self.column_names_to_id = {}
        for i, (table_id, column_name) in enumerate(column_names_original):
            if table_id >= 0:
                table_name = table_names_original[table_id]
                col_name = '{}.{}'.format(table_name,column_name)
            else:
                col_name = column_name
            col_name = col_name.lower()
            if col_name not in self.column_names_to_id:
                self.column_names_surface.append(col_name)
                self.column_names_to_id[col_name] = len(self.column_names_surface) - 1
                column_keep_index.append(i)

        start_i = len(self.column_names_to_id)
        for i, table_name in enumerate(table_names_original):
            col_name = '{}.*'.format(table_name.lower())
            self.column_names_surface.append(col_name)
            self.column_names_to_id[col_name] = i + start_i

        self.column_names_emb_input = []
        self.column_names_emb_input_to_id = {}
        for i, (table_id, column_name) in enumerate(column_names):
            if table_id >= 0:
                table_name = table_names[table_id]
                column_name_embedder_input = table_name + ' . ' + column_name
            else:
                column_name_embedder_input = column_name
            if i in column_keep_index:
                self.column_names_emb_input.append(column_name_embedder_input)
                self.column_names_emb_input_to_id[column_name_embedder_input] = len(self.column_names_emb_input) - 1

        start_i = len(self.column_names_emb_input_to_id)
        for i, table_name in enumerate(table_names):
            column_name_embedder_input = table_name + ' . *'
            self.column_names_emb_input.append(column_name_embedder_input)
            self.column_names_emb_input_to_id[column_name_embedder_input] = i + start_i

        assert len(self.column_names_surface) == len(self.column_names_to_id) == len(self.column_names_emb_input) == len(self.column_names_emb_input_to_id)

        assert len(self.column_names_surface)-1 == \
            max(v for k,v in self.column_names_to_id.items()) == \
            max(v for k,v in self.column_names_emb_input_to_id.items())

        self.num_col = len(self.column_names_surface)

    def __len__(self):
        return self.num_col

    def in_vocab(self, column_name, surface_form=False):
        if surface_form:
            return column_name in self.column_names_to_id
        else:
            return column_name in self.column_names_emb_input_to_id

    def column_name_embedder_bow(self, column_name, surface_form=False, column_name_token_embedder=None):
        assert self.in_vocab(column_name, surface_form)
        if surface_form:
            column_name_id = self.column_names_to_id[column_name]
            column_name_embedder_input = self.column_names_emb_input[column_name_id]
        else:
            column_name_embedder_input = column_name

        column_name_embeddings = [column_name_token_embedder(token) for token in column_name_embedder_input.split()]
        column_name_embeddings = torch.stack(column_name_embeddings, dim=0)
        return torch.mean(column_name_embeddings, dim=0)

    def set_column_name_embeddings(self, column_name_embeddings):
        self.column_name_embeddings = column_name_embeddings
        assert len(self.column_name_embeddings) == self.num_col

    def column_name_embedder(self, column_name, surface_form=False):
        assert self.in_vocab(column_name, surface_form)
        if surface_form:
            column_name_id = self.column_names_to_id[column_name]
        else:
            column_name_id = self.column_names_emb_input_to_id[column_name]

        return self.column_name_embeddings[column_name_id]