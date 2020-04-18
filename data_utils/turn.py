"""Contains the Turn class and token functions."""

import nltk
import sqlparse

# keys of json
INPUT_KEY = 'utterance'
OUTPUT_KEY = 'sql'


def nl_tokenize(string):
    """Tokenizes a natural language string into tokens.
    Assumes data is space-separated (this is true of ZC07 data in ATIS2/3).

    Args:
       string: the string to tokenize.

    Returns:
        a list of tokens.
    """
    return nltk.word_tokenize(string)

def sql_tokenize(string):
    """Tokenizes a SQL statement into tokens.

    Args:
       string: string to tokenize.

    Returns:
       list of str: table.column is treated as a single token.
    """
    tokens = []
    statements = sqlparse.parse(string)

    # SQLparse gives you a list of statements.
    for statement in statements:
        # Flatten the tokens in each statement and add to the tokens list.
        flat_tokens = sqlparse.sql.TokenList(statement.tokens).flatten()
        for token in flat_tokens:
            strip_token = str(token).strip()
            if len(strip_token) > 0:
                tokens.append(strip_token)

    merged_tokens = []
    keep = True
    for i, token in enumerate(tokens):
        if token == ".":
            new_token = merged_tokens[-1] + "." + tokens[i + 1]
            merged_tokens = merged_tokens[:-1] + [new_token]
            keep = False
        elif keep:
            merged_tokens.append(token)
        else:
            keep = True

    return merged_tokens


class Turn:
    """Contains a turn in an interaction.
    
    Attributes:
        utter_seq (list of str): nl tokens.
        query_seq (list of str): sql tokens.
        keep (bool): if not empty.
    """

    def __init__(self, example):
        self.utter_seq = nl_tokenize(example[INPUT_KEY])
        self.query_seq = example[OUTPUT_KEY]
        self.keep = self.query_seq and self.utter_seq

    def __str__(self):
        return 'Input: ' + ' '.join(self.utter_seq) + '\n' + \
            'Output: ' + ' '.join(self.query_seq) + '\n'

    def length_valid(self, utter_limit, query_limit):
        return len(self.utter_seq) < utter_limit \
            and len(self.query_seq) < query_limit
