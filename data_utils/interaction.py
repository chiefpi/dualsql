"""Contains the class for an interaction."""
import torch


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

    def input_seqs(self):
        return [turn.input_seq for turn in self.turns]

    def output_seqs(self):
        return [turn.output_seq for turn in self.turns]

    def set_valid_length(self, max_input_length, max_output_length):
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
