import json
from data_utils.vocab import Vocab

def load_db_schema(db_schema_filename, remove_from=True):
    """Reads schema for the preprocessed dataset.

    Returns:
        db2schema (dict int -> Schema)
        schema_tokens (list of str): Actually column names. For query vocab skip tokens.
        schema_tokens_sep (list of list of str): Actually used.
    """

    with open(db_schema_filename, 'r') as f:
        db_schema = json.load(f)

    db_schema_dict = {}
    all_schema_tokens = []
    all_schema_tokens_sep = []
    for table_schema in db_schema:
        db_id = table_schema['db_id']
        column_names = table_schema['column_names']
        column_names_original = table_schema['column_names_original']
        table_names = table_schema['table_names']
        table_names_original = table_schema['table_names_original']

        schema_tokens = []
        schema_tokens_sep = []
        if remove_from:
            schema_tokens += ['{}.{}'.format(table_names_original[table_id], column_name).lower()
                if table_id >= 0 else column_name.lower()
                for table_id, column_name in column_names_original]
            schema_tokens += ['{}.*'.format(table_name.lower())
                for table_name in table_names_original]

            schema_tokens_sep += ['{} . {}'.format(table_names[table_id], column_name).split()
                if table_id >= 0 else column_name.lower()
                for table_id, column_name in column_names]
            schema_tokens_sep += ['{} . *'.format(table_name).split()
                for table_name in table_names]
        else:
            schema_tokens += [column_name.lower() for _, column_name in column_names_original]
            schema_tokens += [table_name.lower() for table_name in table_names_original]

            schema_tokens_sep += [column_name.split() for _, column_name in column_names]
            schema_tokens_sep += [table_name.split() for table_name in table_names]

        db_schema_dict[db_id] = Schema(schema_tokens_sep, schema_tokens)
        all_schema_tokens += schema_tokens
        all_schema_tokens_sep += schema_tokens_sep

    return db_schema_dict, all_schema_tokens, all_schema_tokens_sep


class Schema:
    """Contains a schema.

    Attributes:
        schema_tokens_sep (list of list of str)
    """
    def __init__(self, schema_tokens_sep, schema_tokens):
        self.type = str
        self.schema_tokens_sep = schema_tokens_sep
        self.vocab = Vocab([schema_tokens])

    def __len__(self):
        return len(self.schema_tokens_sep)

    def str2index(self, schema_vocab):
        if self.type == str:
            self.schema_tokens_sep = [[schema_vocab.token2id[t] for t in token_sep]
                for token_sep in self.schema_tokens_sep]
            self.type = int

    def index2str(self, schema_vocab):
        if self.type == int:
            self.schema_tokens_sep = [[schema_vocab.id2token[i] for i in token_sep]
                for token_sep in self.schema_tokens_sep]
            self.type = str