"""Pretrains the primal model with labeled data."""

import os
import sys
import random
import argparse

import numpy as np
import torch
import torch.nn as nn

from logger import Logger
from data_utils import corpus
from data_utils.corpus import Corpus
from models.dualsql import DualSQL


# Set the random seed manually for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_params():
    parser = argparse.ArgumentParser()
    # Filenames
    parser.add_argument('--embedding_filename', type=str)
    parser.add_argument('--input_vocabulary_filename', type=str,
        default='input_vocabulary.pkl')
    parser.add_argument('--output_vocabulary_filename', type=str,
        default='output_vocabulary.pkl')
    parser.add_argument('--data_dir', type=str,
        default='data/sparc')
    parser.add_argument('--raw_train_filename', type=str,
        default='data/sparc_data_removefrom/train.pkl')
    parser.add_argument('--raw_validation_filename', type=str,
        default='data/sparc_data_removefrom/dev.pkl')
    # Embedder
    parser.add_argument('--use_bert', action='store_true')
    # Decoder
    parser.add_argument('--use_editing', action='store_true')
    parser.add_argument('--max_turns_to_keep', type=int, default=5)
    # Training
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--evaluate_split', choices=['valid', 'dev', 'test'])
    # Logging
    parser.add_argument('--log_dir', type=str, 
        default='logs/semantic_parsing')
    parser.add_argument('--save_file', type=str,
        default='model.pt')

    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if args.train:
        args_file = args.log_dir + '/args.log'
        if os.path.exists(args_file):
            raise ValueError('Warning: arguments already exist in ' + str(args_file))
        with open(args_file, 'w') as infile:
            infile.write(str(args))

    return args

def train_step(model, interaction, params):
    """Trains the model on a single interation."""
    schema = interaction.schema
    for index, turn in enumerate(interaction.turns):
        input_seq = turn.input_seq
        prev_queries = interaction.prev_queries(index)
        gold_query = turn.output_seq


def train(model, data, params):
    pass


def evaluate(model, data, params, split):
    pass


if __name__ == "__main__":
    params = get_params()

    data = Corpus(params)
    model = DualSQL(
        data.input_vocab,
        data.output_vocab,
        data.output_vocab_schema,
        params.embedding_filename,
        freeze=params.freeze,
        dropout=params.dropout,
        use_editing=params.use_editing,
        max_turns_to_keep=params.max_turns_to_keep,
        use_bert=params.use_bert).to(device)

    print('=====================Model Parameters=====================')
    for name, param in model.named_parameters():
        print(name, param.requires_grad, param.is_cuda, param.size())
        assert param.is_cuda

    model.build_optim()

    print('=====================Parameters in Optimizer==============')
    for param_group in model.trainer.param_groups:
        print(param_group.keys())
        for param in param_group['params']:
            print(param.size())

    # if params.fine_tune_bert:
    #     print('=====================Parameters in BERT Optimizer==============')
    #     for param_group in model.bert_trainer.param_groups:
    #         print(param_group.keys())
    #         for param in param_group['params']:
    #             print(param.size())

    sys.stdout.flush()

    if params.train:
        train(model, data, params)
    if params.evaluate and 'valid' in params.evaluate_split:
        evaluate(model, data, params, split='valid')
    if params.evaluate and 'dev' in params.evaluate_split:
        evaluate(model, data, params, split='dev')
    if params.evaluate and 'test' in params.evaluate_split:
        evaluate(model, data, params, split='test')