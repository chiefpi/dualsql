"""Contains class and methods for storing and computing a vocabulary from text."""
import operator

# Special sequencing tokens.
BOS_TOK = '<bos>' # Marks the beginning of generation.
UNK_TOK = '<unk>' # Replaces out-of-vocabulary tokens.
EOS_TOK = '<eos>' # Appended to the end of a sequence to indicate its end.
PAD_TOK = '<pad>'
DEL_TOK = ';'


class Vocab:
    """Contains a dictionary.

    Attributes:
        id2token (list of str): List of token types.
        token2id (dict str->int): Maps from each unique token type to its index.
    """

    def __init__(self, seqs, data_type=None, skip=[]):
        self.id2token = []
        self.token2id = {}

        if data_type == 'utter':
            functional_types = [PAD_TOK, UNK_TOK, BOS_TOK, EOS_TOK, DEL_TOK]
        elif data_type == 'query':
            functional_types = [PAD_TOK, UNK_TOK, BOS_TOK, EOS_TOK]
        elif data_type == 'schema':
            functional_types = [PAD_TOK, UNK_TOK]
        else:
            functional_types = []

        for token in functional_types:
            self.add_token(token)

        for seq in seqs:
            for token in seq:
                if token not in skip:
                    self.add_token(token)

    def add_token(self, token):
        if token not in self.token2id:
            self.id2token.append(token)
            self.token2id[token] = len(self.id2token) - 1
        return self.token2id[token]

    def __len__(self):
        return len(self.id2token)
