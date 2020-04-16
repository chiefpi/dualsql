# TODO: rewrite when I need batch

import copy
import math

from data_utils.vocab import EOS_TOK


def fix_parentheses(seq):
    """Balances parentheses for a sequence.

    Args:
        seq (list of str): Contains an EOS token.
    """
    return seq[:-1] + list(")" * (seq.count("(")-seq.count(")"))) + [seq[-1]]

def flatten_sequence(seq):
    """Removes EOS token from a sequence."""
    
    return fix_parentheses(seq[:-1] if seq[-1] == EOS_TOK else seq)

class TurnItem():
    """Contains a turn."""
    def __init__(self, interaction, index):
        self.turn = interaction[index]
        self.prev_turn = interaction[index-1] if index > 0 else []

    def __str__(self):
        return str(self.turn)

    def input_sequence(self):
        return self.turn.input_seq

    def original_gold_queries(self):
        return [q[0] for q in self.turn.output_seq]

    def gold_tables(self):
        return [q[1] for q in self.turn.output_seq]

    def gold_query(self):
        return self.turn.output_seq + [EOS_TOK]

    def gold_edit_sequence(self):
        return self.turn.gold_edit_sequence

    def gold_table(self):
        return self.turn.gold_sql_results

    def within_limits(
            self,
            max_input_length=math.inf,
            max_output_length=math.inf):
        return self.turn.length_valid(
            max_input_length,
            max_output_length)


class TurnBatch():
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def start(self):
        self.index = 0

    def next(self):
        item = self.items[self.index]
        self.index += 1
        return item

    def done(self):
        return self.index >= len(self.items)


class TurnItemWithPred():
    """Contains a turn and predictions.
    
    Attributes:
        input_seq (list of str)
        pred_query (list of str)
        prev_pred_query (list of str)
    """
    def __init__(
            self,
            input_sequence,
            previous_query): # TODO: previous_queries for dual
        self.input_seq = input_sequence
        self.prev_pred_query = previous_query

    def set_predicted_query(self, query):
        self.pred_query = query


class InteractionItem():
    """Contains an interaction."""

    def __init__(
            self,
            interaction,
            max_input_length=math.inf,
            max_output_length=math.inf,
            max_length=math.inf):
        if max_length != math.inf:
            self.interaction = copy.deepcopy(interaction)
            self.turns = self.interaction.turns[:max_length]
        else:
            self.interaction = interaction
        self.processed_turns = []
        self.identifier = self.interaction.identifier

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self.index = 0

    def __len__(self):
        return len(self.interaction)

    def __str__(self):
        s = 'Turns, gold queries, and predictions:\n'
        for i, turn in enumerate(self.turns):
            s += ' '.join(turn.input_seq) + '\n'
            pred_turn = self.processed_turns[i]
            s += ' '.join(pred_turn.gold_query()) + '\n'
            s += ' '.join(pred_turn.anonymized_query()) + '\n'
            s += '\n'

        return s

    def start_interaction(self):
        assert len(self.processed_turns) == 0
        assert self.index == 0

    def next_turn(self):
        turn = self.turns[self.index]
        self.index += 1

        return PredTurnItem(
            turn.input_seq,
            self,
            self.processed_turns[-1].pred_query if len(self.processed_turns) > 0 else [],
            self.index-1)

    def done(self):
        return len(self.processed_turns) == len(self.interaction)

    def finish(self):
        self.processed_turns = []
        self.index = 0

    def turn_within_limits(self, turn_item):
        return turn_item.within_limits(
            self.max_input_length,
            self.max_output_length)

    def gold_turns(self):
        turns = []
        for i, turn in enumerate(self.turns):
            turns.append(TurnItem(self.interaction, i))
        return turns

    def get_schema(self):
        return self.interaction.schema

    def add_turn(self, turn, predicted_sequence, simple=False):
        self.processed_turns.append(turn)



    def gold_query(self, index):
        return self.turns[index].output_seq + [EOS_TOK]

    def original_gold_query(self, index):
        return self.turns[index].original_gold_query

    def gold_table(self, index):
        return self.turns[index].gold_sql_results
