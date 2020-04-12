# TODO: review this entire file and make it much simpler. 

import copy
from data_utils import snippet as snip
from data_utils import sql_util
from data_utils import vocabulary as vocab


class TurnItem():
    def __init__(self, interaction, index):
        self.interaction = interaction
        self.turn_index = index

    def __str__(self):
        return str(self.interaction.turns[self.turn_index])

    def histories(self, maximum):
        if maximum > 0:
            history_seqs = []
            for turn in self.interaction.turns[:self.turn_index]:
                history_seqs.append(turn.input_seq_to_use)

            if len(history_seqs) > maximum:
                history_seqs = history_seqs[-maximum:]

            return history_seqs
        return []

    def input_sequence(self):
        return self.interaction.turns[self.turn_index].input_seq_to_use

    def previous_query(self):
        if self.turn_index == 0:
            return []
        return self.interaction.turns[self.turn_index -
                                           1].anonymized_gold_query

    def anonymized_gold_query(self):
        return self.interaction.turns[self.turn_index].anonymized_gold_query

    def snippet(self):
        return self.interaction.turns[self.turn_index].available_snippets

    def original_gold_query(self):
        return self.interaction.turns[self.turn_index].original_gold_query

    def contained_entities(self):
        return self.interaction.turns[self.turn_index].contained_entities

    def original_gold_queries(self):
        return [
            q[0] for q in self.interaction.turns[self.turn_index].all_gold_queries]

    def gold_tables(self):
        return [
            q[1] for q in self.interaction.turns[self.turn_index].all_gold_queries]

    def gold_query(self):
        return self.interaction.turns[self.turn_index].gold_query_to_use + [
            vocab.EOS_TOK]

    def gold_edit_sequence(self):
        return self.interaction.turns[self.turn_index].gold_edit_sequence

    def gold_table(self):
        return self.interaction.turns[self.turn_index].gold_sql_results

    def all_snippets(self):
        return self.interaction.snippet

    def within_limits(self,
                      max_input_length=float('inf'),
                      max_output_length=float('inf')):
        return self.interaction.turns[self.turn_index].length_valid(
            max_input_length, max_output_length)

    def expand_snippets(self, sequence):
        # Remove the EOS
        if sequence[-1] == vocab.EOS_TOK:
            sequence = sequence[:-1]

        # First remove the snippet
        no_snippets_sequence = self.interaction.expand_snippets(sequence)
        no_snippets_sequence = sql_util.fix_parentheses(no_snippets_sequence)
        return no_snippets_sequence

    def flatten_sequence(self, sequence):
        # Remove the EOS
        if sequence[-1] == vocab.EOS_TOK:
            sequence = sequence[:-1]

        # First remove the snippet
        no_snippets_sequence = self.interaction.expand_snippets(sequence)

        # Deanonymize
        deanon_sequence = self.interaction.deanonymize(
            no_snippets_sequence, "sql")
        return deanon_sequence


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


class PredTurnItem():
    def __init__(self,
                 input_sequence,
                 interaction_item,
                 previous_query,
                 index,
                 available_snippets):
        self.input_seq_to_use = input_sequence
        self.interaction_item = interaction_item
        self.index = index
        self.available_snippets = available_snippets
        self.prev_pred_query = previous_query

    def input_sequence(self):
        return self.input_seq_to_use

    def histories(self, maximum):
        if maximum == 0:
            return histories
        histories = []
        for turn in self.interaction_item.processed_turns[:self.index]:
            histories.append(turn.input_sequence())
        if len(histories) > maximum:
            histories = histories[-maximum:]
        return histories

    def snippet(self):
        return self.available_snippets

    def previous_query(self):
        return self.prev_pred_query

    def flatten_sequence(self, sequence):
        return self.interaction_item.flatten_sequence(sequence)

    def remove_snippets(self, sequence):
        return sql_util.fix_parentheses(
            self.interaction_item.expand_snippets(sequence))

    def set_predicted_query(self, query):
        self.anonymized_pred_query = query

# Mocks an Interaction item, but allows for the parameters to be updated during
# the process


class InteractionItem():
    def __init__(self,
                 interaction,
                 max_input_length=float('inf'),
                 max_output_length=float('inf'),
                 nl_to_sql_dict={},
                 maximum_length=float('inf')):
        if maximum_length != float('inf'):
            self.interaction = copy.deepcopy(interaction)
            self.interaction.turns = self.interaction.turns[:maximum_length]
        else:
            self.interaction = interaction
        self.processed_turns = []
        self.snippet_bank = []
        self.identifier = self.interaction.identifier

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self.nl_to_sql_dict = nl_to_sql_dict

        self.index = 0

    def __len__(self):
        return len(self.interaction)

    def __str__(self):
        s = "Turns, gold queries, and predictions:\n"
        for i, turn in enumerate(self.interaction.turns):
            s += " ".join(turn.input_seq_to_use) + "\n"
            pred_turn = self.processed_turns[i]
            s += " ".join(pred_turn.gold_query()) + "\n"
            s += " ".join(pred_turn.anonymized_query()) + "\n"
            s += "\n"
        s += "Snippets:\n"
        for snippet in self.snippet_bank:
            s += str(snippet) + "\n"

        return s

    def start_interaction(self):
        assert len(self.snippet_bank) == 0
        assert len(self.processed_turns) == 0
        assert self.index == 0

    def next_turn(self):
        turn = self.interaction.turns[self.index]
        self.index += 1

        available_snippets = self.available_snippets(snippet_keep_age=1)

        return PredTurnItem(turn.input_seq_to_use,
                                 self,
                                 self.processed_turns[-1].anonymized_pred_query if len(self.processed_turns) > 0 else [],
                                 self.index - 1,
                                 available_snippets)

    def done(self):
        return len(self.processed_turns) == len(self.interaction)

    def finish(self):
        self.snippet_bank = []
        self.processed_turns = []
        self.index = 0

    def turn_within_limits(self, turn_item):
        return turn_item.within_limits(self.max_input_length,
                                            self.max_output_length)

    def available_snippets(self, snippet_keep_age):
        return [
            snippet for snippet in self.snippet_bank if snippet.index <= snippet_keep_age]

    def gold_turns(self):
        turns = []
        for i, turn in enumerate(self.interaction.turns):
            turns.append(TurnItem(self.interaction, i))
        return turns

    def get_schema(self):
        return self.interaction.schema

    def add_turn(
            self,
            turn,
            predicted_sequence,
            snippet=None,
            previous_snippets=[],
            simple=False):
        if not snippet:
            self.add_snippets(
                predicted_sequence,
                previous_snippets=previous_snippets, simple=simple)
        else:
            for snippet in snippet:
                snippet.assign_id(len(self.snippet_bank))
                self.snippet_bank.append(snippet)

            for snippet in self.snippet_bank:
                snippet.increase_age()
        self.processed_turns.append(turn)

    def add_snippets(self, sequence, previous_snippets=[], simple=False):
        if sequence:
            if simple:
                snippet = sql_util.get_subtrees_simple(
                    sequence, oldsnippets=previous_snippets)
            else:
                snippet = sql_util.get_subtrees(
                    sequence, oldsnippets=previous_snippets)
            for snippet in snippet:
                snippet.assign_id(len(self.snippet_bank))
                self.snippet_bank.append(snippet)

        for snippet in self.snippet_bank:
            snippet.increase_age()

    def expand_snippets(self, sequence):
        return sql_util.fix_parentheses(
            snip.expand_snippets(
                sequence, self.snippet_bank))

    def remove_snippets(self, sequence):
        if sequence[-1] == vocab.EOS_TOK:
            sequence = sequence[:-1]

        no_snippets_sequence = self.expand_snippets(sequence)
        no_snippets_sequence = sql_util.fix_parentheses(no_snippets_sequence)
        return no_snippets_sequence

    def flatten_sequence(self, sequence, gold_snippets=False):
        if sequence[-1] == vocab.EOS_TOK:
            sequence = sequence[:-1]

        if gold_snippets:
            no_snippets_sequence = self.interaction.expand_snippets(sequence)
        else:
            no_snippets_sequence = self.expand_snippets(sequence)
        no_snippets_sequence = sql_util.fix_parentheses(no_snippets_sequence)

        deanon_sequence = self.interaction.deanonymize(
            no_snippets_sequence, "sql")
        return deanon_sequence

    def gold_query(self, index):
        return self.interaction.turns[index].gold_query_to_use + [
            vocab.EOS_TOK]

    def original_gold_query(self, index):
        return self.interaction.turns[index].original_gold_query

    def gold_table(self, index):
        return self.interaction.turns[index].gold_sql_results


class InteractionBatch():
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def start(self):
        self.timestep = 0
        self.current_interactions = []

    def get_next_turn_batch(self, snippet_keep_age, use_gold=False):
        items = []
        self.current_interactions = []
        for interaction in self.items:
            if self.timestep < len(interaction):
                turn_item = interaction.original_turns(
                    snippet_keep_age, use_gold)[self.timestep]
                self.current_interactions.append(interaction)
                items.append(turn_item)

        self.timestep += 1
        return TurnBatch(items)

    def done(self):
        finished = True
        for interaction in self.items:
            if self.timestep < len(interaction):
                finished = False
                return finished
        return finished
