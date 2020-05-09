"""Trains and evaluates a language model for validity rewards."""

import time
import math
import os
import sys
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_utils.logger import Logger
from data_utils import corpus
from data_utils.corpus import Corpus
from models.language_model import LanguageModel


# Set the random seed manually for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_params():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--input_vocab_filename', type=str,
        default='input_vocabulary.pkl')
    parser.add_argument('--output_vocab_filename', type=str,
        default='output_vocabulary.pkl')
    parser.add_argument('--data_dir', type=str,
        default='processed_sparc_data_removefrom')

    parser.add_argument('--raw_train_filename', type=str,
        default='data/sparc_data_removefrom/train.pkl')
    parser.add_argument('--raw_valid_filename', type=str,
        default='data/sparc_data_removefrom/dev.pkl')
    parser.add_argument('--raw_test_filename', type=str,
        default='data/sparc_data_removefrom/test.pkl')
    
    parser.add_argument('--processed_train_filename', type=str,
        default='train.pkl')
    parser.add_argument('--processed_dev_filename', type=str,
        default='dev.pkl')
    parser.add_argument('--processed_valid_filename', type=str,
        default='validation.pkl')
    parser.add_argument('--processed_test_filename', type=str,
        default='test.pkl')

    parser.add_argument('--database_schema_filename', type=str,
        default='data/sparc_data_removefrom/tables.json')
    parser.add_argument('--embedding_filename', type=str,
        default='glove/glove.840B.300d.txt')

    # Model
    parser.add_argument('--primal', type=bool, default=False)
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--tied', action='store_true',
        help='tie the word embedding and softmax weights')
    # Training
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=20)
    parser.add_argument('--clip', type=float, default=0.25)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--train_eval_size', type=int, default=20)
    parser.add_argument('--evaluate_split', choices=['valid', 'dev', 'test'])
    # Logging
    parser.add_argument('--log_name', type=str, 
        default='lm')
    parser.add_argument('--save_file', type=str,
        default='model.pt')

    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if args.train:
        args_file = os.path.join(args.log_dir, 'args.log')
        if os.path.exists(args_file):
            raise ValueError('Warning: arguments already exist in {}'.format(args_file))
        with open(args_file, 'w') as infile:
            infile.write(str(args))

    return args


def train_batches(model, batches, optimizer, criterion, params):
    model.train()
    epoch_loss = 0.
    for batch in batches:
        optimizer.zero_grad()
        for turn in batch:
            sent = turn.utter_seq if params.primal else turn.query_seq
            output = model(sent[:-1])
            loss = criterion(output, sent[1:])
            loss.backward() # TODO: batch loss

        # Gradient clip
        nn.utils.clip_grad_norm_(model.parameters(), params.clip)

        epoch_loss += loss.item()
        optimizer.step()

    return epoch_loss


def eval_turns(model, turns, criterion, params):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for turn in turns:
            sent = turn.utter_seq if params.primal else turn.query_seq
            output = model(sent[:-1])
            total_loss += criterion(output, sent[1:]).item()

    return total_loss / len(turns)
    

def train(model, data, params):
    """Trains a language model on a corpus."""

    log = Logger(20, params.log_name)
    num_train = corpus.num_turns(data.train_data)
    log.info('Total number of training turns: {:d}' % num_train)

    train_batches = data.get_turn_batches(params.batch_size)
    # evaluation samples
    train_samples = data.get_random_turns(params.train_eval_size)
    valid_samples = data.get_all_turns(data.valid_data)

    log.info('Number of steps per epoch: {:d}' % len(train_batches))
    log.info('Batch size: {:d}' % params.batch_size)
    assert params.batch_size == 1 # TODO

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    best_valid_loss = None
    # Loop over epochs.
    for epoch in range(params.epochs):
        log.info('Epoch: {:d}' % epoch)
        epoch_loss = train_batches(
            model, train_batches, optimizer, criterion, params)
        log.info('Train epoch loss: {:.3f}'.format(epoch_loss/num_train))

        train_eval_loss = eval_turns(
            model, train_samples, criterion, params)
        log.info('Train evaluation loss: {:.3f} | ppl: {:.3f}'.format(
            train_eval_loss, math.exp(train_eval_loss)))

        valid_loss = eval_turns(
            model, valid_samples, criterion, params)
        log.info('Validation loss: {:.3f} | ppl: {:.3f}'.format(
            valid_loss, math.exp(valid_loss)))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or valid_loss < best_valid_loss:
            model.save(os.path.join(params.log_dir, params.save_file))
            best_val_loss = valid_loss

    log.info('Finished training!')


def evaluate(model, data, params, split):
    """Evaluates a language model on a corpus.

    Args:
        model (LanguageModel)
        data (Corpus)
        params (namespace)
        split (str): Split to evaluate (valid, dev or test).
    """
    model.load(os.path.join(params.log_dir, params.language_model_file))
    data_split = eval('data.{}_data'.format(split))
    examples = data.get_all_turns(data_split)
    criterion = nn.NLLLoss()
    
    eval_loss = eval_turns(model, examples, criterion, params)
    print('Evaluation loss: {:.3f} | ppl: {:.3f}'.format(
        eval_loss, math.exp(eval_loss)))


if __name__ == '__main__':
    params = get_params()

    data = Corpus(params)
    vocab = data.input_vocab if params.primal else data.output_vocab
    model = LanguageModel(
        vocab=vocab,
        emb_file=params.embedding_filename,
        hidden_dim=params.hidden_dim,
        num_layers=params.num_layers,
        freeze=params.primal,
        dropout=params.dropout,
        tie_weights=params.tied).to(device)

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

    sys.stdout.flush()

    if params.train:
        train(model, data, params)
    if params.evaluate and 'valid' in params.evaluate_split:
        evaluate(model, data, params, split='valid')
    if params.evaluate and 'dev' in params.evaluate_split:
        evaluate(model, data, params, split='dev')
    if params.evaluate and 'test' in params.evaluate_split:
        evaluate(model, data, params, split='test')
