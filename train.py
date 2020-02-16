import os
import sys

import numpy as np
import random
import torch

import argparse
import data_util

from models.sqlnet.scripts.model import SQLNet
from models.cdseq2seq.scripts import CDSeq2Seq
# from models.editsql.scripts import EditSQL


# fix seeds
np.random.seed(0)
random.seed(0)


def get_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', choices=('spider', 'sparc', 'cosql'), default='sparc')
    parser.add_argument('--model', choices=('sqlnet', 'cdseq2seq'), default='cdseq2seq')

    return parser.parse_args()


def load_dataset(dataset_name):
    """
    """
    # TODO: anonymizer

    # read database schema
    # TODO: why is simple, what is remove_from
    def read_database_schema(self, database_schema_filename):
        with open(database_schema_filename, "r") as f:
            database_schema = json.load(f)

        database_schema_dict = {}
        column_names_surface_form = []
        column_names_embedder_input = []
        for table_schema in database_schema:
            db_id = table_schema['db_id']
            database_schema_dict[db_id] = table_schema

            column_names = table_schema['column_names']
            column_names_original = table_schema['column_names_original']
            table_names = table_schema['table_names']
            table_names_original = table_schema['table_names_original']

            for i, (table_id, column_name) in enumerate(column_names_original):
                column_name_surface_form = column_name
                column_names_surface_form.append(column_name_surface_form.lower())

            for table_name in table_names_original:
                column_names_surface_form.append(table_name.lower())

            for i, (table_id, column_name) in enumerate(column_names):
                column_name_embedder_input = column_name
                column_names_embedder_input.append(column_name_embedder_input.split())

            for table_name in table_names:
                column_names_embedder_input.append(table_name.split())

        database_schema = database_schema_dict

        return database_schema, column_names_surface_form, column_names_embedder_input

    database_schema, column_names_surface_form, column_names_embedder_input = read_database_schema(params.database_schema_filename)

    # collapse a list of list into a single list
    collapse_list = lambda x:[s for i in x for s in i]

    train_data = DatasetSplit(
        os.path.join(params.data_directory, params.processed_train_filename),
        params.raw_train_filename,
        int_load_function
    )
    valid_data = DatasetSplit(

    )

    train_input_seqs = collapse_list(train_data.get_ex_properties(l))
    valid_input_seqs = collapse_list(train_data.get_ex_properties(l))

    return data


def load_model(model_name):
    if model_name == 'sqlnet':
        model = SQLNet()
    elif model_name == 'cdseq2seq':
        model = CDSeq2Seq()
    # elif model_name == 'editsql':
    #     model = EditSQL()
    
    return model


def train(model, data):
    # TODO: log

    
    return


if __name__ == "__main__":
    params = get_params()

    data = load_dataset(params.dataset)

    model = load_model(params.model)
    model = model.cuda()

    train(model, data)