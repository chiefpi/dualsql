import os
import pickle

from data_utils.schema import Schema
from data_utils.turn import Turn
from data_utils.interaction import Interaction

def collapse_list(lst):
    """Collapses a list of list into a single list."""
    return [i for l in lst for i in l]

class DatasetSplit:
    """Stores a split of the dataset.

    Attributes:
        interactions (list of Interaction): Stores the interactions in the split.
    """
    def __init__(self, processed_filename, raw_filename, db2schema):
        if os.path.exists(processed_filename):
            print("Loading preprocessed data from " + processed_filename)
            with open(processed_filename, 'rb') as infile:
                self.interactions = pickle.load(infile)
        else:
            print("Loading raw data from " + raw_filename +
                " and writing to " + processed_filename)

            with open(raw_filename, 'rb') as infile:
                examples_from_file = pickle.load(infile)
                assert isinstance(examples_from_file, list), raw_filename + \
                    " does not contain a list of interactions"

            self.interactions = []
            for example in examples_from_file:
                interaction, keep = self.load_interaction(example, db2schema)
                if keep:
                    self.interactions.append(interaction)

            print("Loaded " + str(len(self.interactions)) + " interactions")
            with open(processed_filename, 'wb') as outfile:
                pickle.dump(self.interactions, outfile)

    def load_interaction(self, interaction_example, db2schema):
        """Loads an example to Interaction.

        Returns:
            interaction (Interaction)
            keep (bool): Keep if not empty.
        """

        raw_turns = interaction_example['interaction']

        database_id = interaction_example['database_id']
        interaction_id = interaction_example['interaction_id']
        identifier = '{}/{}'.format(database_id, interaction_id)

        schema = db2schema[database_id]

        keep = False
        turns = []

        for raw_turn in raw_turns:
            turn = Turn(raw_turn)
            keep_turn = turn.keep
            assert not schema or keep_turn # if schema, then keep
            if keep_turn:
                keep = True
                turns.append(turn)

        interaction = Interaction(turns, schema, identifier)

        return interaction, keep

    def __len__(self):
        """Returns the number of turns in a data split."""
        return sum([len(i) for i in self.interactions])

    def get_all_utterances(self):
        """
        Returns:
            list of list of str
        """
        return collapse_list([i.utter_seqs() for i in self.interactions])

    def get_all_utterances_id(self):
        """
        Returns:
            list of list of int
        """
        return collapse_list([i.utter_seqs_id() for i in self.interactions])

    def get_all_queries(self):
        return collapse_list([i.query_seqs() for i in self.interactions])

    def get_all_queries_id(self):
        return collapse_list([i.query_seqs_id() for i in self.interactions])

    def str2index(self, schema_vocab, utter_vocab, query_vocab):
        for interaction in self.interactions:
            interaction.str2index(schema_vocab, utter_vocab, query_vocab)
        
    def index2str(self, schema_vocab, utter_vocab, query_vocab):
        for interaction in self.interactions:
            interaction.index2str(schema_vocab, utter_vocab, query_vocab)
