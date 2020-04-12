"""Tokenizers for natural language and SQL queries."""
import nltk
import sqlparse

def nl_tokenize(string):
    """Tokenizes a natural language string into tokens.

    Inputs:
       string: the string to tokenize.
    Outputs:
        a list of tokens.

    Assumes data is space-separated (this is true of ZC07 data in ATIS2/3).
    """
    return nltk.word_tokenize(string)

def sql_tokenize(string):
    """Tokenizes a SQL statement into tokens.

    Inputs:
       string: string to tokenize.

    Outputs:
       a list of tokens.
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

    newtokens = []
    keep = True
    for i, token in enumerate(tokens):
        if token == ".":
            newtoken = newtokens[-1] + "." + tokens[i + 1]
            newtokens = newtokens[:-1] + [newtoken]
            keep = False
        elif keep:
            newtokens.append(token)
        else:
            keep = True

    return newtokens
