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
        return '\n'.join(['Turns:'] + [str(turn) for turn in self.turns]) + '\n'

    def __len__(self):
        return len(self.turns)

    def utter_seqs(self):
        return [turn.utter_seq for turn in self.turns]

    def query_seqs(self):
        return [turn.query_seq for turn in self.turns]

    def length_valid(self, max_utter_length, max_query_length):
        for turn in self.turns:
            if not turn.length_valid(max_utter_length, max_query_length):
                return False
        return True

    def str2index(self, schema_vocab, utter_vocab, query_vocab):
        self.schema.str2index(schema_vocab)
        for turn in self.turns:
            turn.str2index(self.schema, utter_vocab, query_vocab)

    def index2str(self, schema_vocab, utter_vocab, query_vocab):
        self.schema.index2str(schema_vocab)
        for turn in self.turns:
            turn.index2str(self.schema, utter_vocab, query_vocab)