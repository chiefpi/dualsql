import os
import pickle

from data_utils.schema import Schema
from data_utils.turn import Turn
from data_utils.interaction import Interaction


def load_function(db_schema=None, remove_from=True):
    def fn(interaction_example):
        """Loads an example to Interaction.

        Returns:
            interaction (Interaction)
            keep (bool): Keep if not empty.
        """

        raw_turns = interaction_example['interaction']

        database_id = interaction_example['database_id']
        interaction_id = interaction_example['interaction_id']
        identifier = str(database_id) + '/' + str(interaction_id)

        schema = None
        if db_schema:
            schema = Schema(db_schema[database_id], simple=not remove_from)

        keep = False
        turns = []

        for raw_turn in raw_turns:
            turn = Turn(raw_turn)
            keep_turns = turn.keep
            assert not schema or keep_turns # if schema, then keep
            if keep_turns:
                keep = True
                turns.append(turn)

        interaction = Interaction(turns, schema, identifier)

        return interaction, keep

    return fn


class DatasetSplit:
    """Stores a split of the dataset.

    Attributes:
        examples (list of Interaction): Stores the examples in the split.
    """
    def __init__(self, processed_filename, raw_filename, load_function):
        if os.path.exists(processed_filename):
            print("Loading preprocessed data from " + processed_filename)
            with open(processed_filename, 'rb') as infile:
                self.examples = pickle.load(infile)
        else:
            print("Loading raw data from " + raw_filename +
                " and writing to " + processed_filename)

            with open(raw_filename, 'rb') as infile:
                examples_from_file = pickle.load(infile)
                assert isinstance(examples_from_file, list), raw_filename + \
                    " does not contain a list of examples"

            self.examples = []
            for example in examples_from_file:
                interaction, keep = load_function(example)
                if keep:
                    self.examples.append(interaction)

            print("Loaded " + str(len(self.examples)) + " examples")
            with open(processed_filename, 'wb') as outfile:
                pickle.dump(self.examples, outfile)

    def get_ex_properties(self, function):
        """Applies some function to the examples in the dataset.

        Args:
            function: (lambda Interaction -> T): Function to apply to all
                examples.

        Returns
            list of the return value of the function
        """
        return [function(ex) for ex in self.examples]
