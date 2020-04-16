import os
import pickle

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
            print(
                "Loading raw data from " +
                raw_filename +
                " and writing to " +
                processed_filename)

            with open(raw_filename, 'rb') as infile:
                examples_from_file = pickle.load(infile)
                assert isinstance(examples_from_file, list), raw_filename + \
                    " does not contain a list of examples"

            self.examples = []
            for example in examples_from_file:
                obj, keep = load_function(example)
                if keep:
                    self.examples.append(obj)

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
