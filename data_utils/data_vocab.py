"""Gets and stores vocabulary for different data types."""

from data_utils.vocab import Vocabulary, UNK_TOK, DEL_TOK, EOS_TOK

MIN_INPUT_OCCUR = 1
MIN_OUTPUT_OCCUR = 1

class DataVocab:
    """Contains vocabulary for data.

    Attributes:
        raw_vocab (Vocabulary): Vocabulary object.
        tokens (set of str): Set of all of the strings in the vocabulary.
        inorder_tokens (list of str): List of all tokens, with a strict and
            unchanging order.
    """
    def __init__(
            self,
            token_sequences,
            filename,
            params,
            data_type='input',
            min_occur=1,
            skip=None):

        if data_type == 'input':
            functional_types = [UNK_TOK, DEL_TOK, EOS_TOK]
        elif data_type == 'output':
            functional_types = [UNK_TOK, EOS_TOK]
        elif data_type == 'schema':
            functional_types = [UNK_TOK]
        else:
            functional_types = []

        self.raw_vocab = Vocabulary(
            token_sequences,
            filename,
            functional_types=functional_types,
            min_occur=min_occur,
            ignore_fn=lambda x: skip and x in skip)
        self.tokens = set(self.raw_vocab.token_to_id.keys())
        self.inorder_tokens = self.raw_vocab.id_to_token

        assert len(self.inorder_tokens) == len(self.raw_vocab)

    def __len__(self):
        return len(self.raw_vocab)

    def token_to_id(self, token):
        """Maps from a token to a unique ID.

        Args:
            token (str): The token to look up.

        Returns:
            int, uniquely identifying the token.
        """
        return self.raw_vocab.token_to_id[token]

    def id_to_token(self, identifier):
        """Maps from a unique integer to an identifier.

        Args:
            identifier (int): The unique ID.

        Returns:
            string, representing the token.
        """
        return self.raw_vocab.id_to_token[identifier]
