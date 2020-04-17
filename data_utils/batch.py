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


class TurnBatch:
    """Contains a turn batch for language models."""
    # TODO: padding
    pass

# class InteractionItem:
#     """Contains an interaction for training.
    
#     Attributes:
#         interaction (Interaction)
#     """

#     def __init__(
#             self,
#             interaction,
#             max_input_length=math.inf,
#             max_output_length=math.inf,
#             max_turns=math.inf):
#         if max_turns != math.inf:
#             self.interaction = copy.deepcopy(interaction) # decouple
#             self.interaction.turns = self.interaction.turns[:max_turns]
#         else:
#             self.interaction = interaction

#         self.max_input_length = max_input_length
#         self.max_output_length = max_output_length

#         self.pred_turns = []
#         self.index = 0

#     def __len__(self):
#         return len(self.interaction)

#     def __str__(self):
#         assert self.done()
#         s = 'Utterances, gold queries, and predictions:\n'
#         for i, turn in enumerate(self.interaction.turns):
#             s += ' '.join(turn.input_seq) + '\n'
#             s += ' '.join(self.processed_turns[i].gold_query) + '\n'
#             s += ' '.join(self.processed_turns[i].pred_query) + '\n'
#             s += '\n'

#         return s

#     def next_turn(self):
#         turn = self.interaction.turns[self.index]
#         self.index += 1

#         return turn

#     def done(self):
#         return len(self.processed_turns) == len(self.interaction)

#     def finish(self):
#         self.processed_turns = []
#         self.index = 0

#     def turn_within_limits(self, turn_item):
#         return turn_item.within_limits(
#             self.max_input_length,
#             self.max_output_length)

#     def gold_turns(self):
#         turns = []
#         for i, turn in enumerate(self.turns):
#             turns.append(TurnItem(self.interaction, i))
#         return turns

#     def get_schema(self):
#         return self.interaction.schema

#     def add_turn(self, turn, predicted_sequence, simple=False):
#         self.processed_turns.append(turn)



#     def gold_query(self, index):
#         return self.turns[index].output_seq + [EOS_TOK]

#     def original_gold_query(self, index):
#         return self.turns[index].original_gold_query

#     def gold_table(self, index):
#         return self.turns[index].gold_sql_results
