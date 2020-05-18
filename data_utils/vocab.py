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

    def __init__(self, seqs, data_type, skip=[]):

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

    # def get_vocab(self, sequences, skip):
    #     """Gets vocabulary from a list of sequences.

    #     Inputs:
    #         sequences (list of list of str): Sequences from which to compute the vocabulary.
    #         skip (list of str): Tokens to skip.

    #     Returns:
    #         list of str, representing the unique token types in the vocabulary.
    #     """
    #     token_count = {}
    #     for sequence in sequences:
    #         for token in sequence:
    #             if token not in skip:
    #                 if token not in token_count:
    #                     token_count[token] = 0
    #                 token_count[token] += 1

    #     # Create sorted list of tokens, by their counts.
    #     # Reverse so it is in order of most frequent to least frequent.
    #     sorted_count = sorted(sorted(token_count.items()),
    #         key=operator.itemgetter(1), reverse=True)

    #     vocab = [token for token, count in sorted_count
    #         if count >= self.min_occur]

    #     # Append the necessary functional tokens.
    #     return vocab + self.functional_types