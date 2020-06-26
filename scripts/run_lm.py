"""Trains and evaluates a language model for validity rewards."""

import math
import os
import sys
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_utils.logger import Logger
from data_utils.batch import get_seq_batches
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
    parser.add_argument('--data_dir', type=str,
        default='processed_sparc_data_removefrom')
    parser.add_argument('--raw_data_dir', type=str,
        default='data/sparc_data_removefrom')
    parser.add_argument('--remove_from', action='store_true')

    parser.add_argument('--db_schema_filename', type=str,
        default='tables.json')
    parser.add_argument('--embedding_filename', type=str,
        default='glove/glove.840B.300d.txt')

    # Model
    parser.add_argument('--primal', action='store_true')
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--tie_weights', action='store_true')
    # Training
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--clip', type=float, default=0.25)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--evaluate_split', choices=['valid', 'dev', 'test'])
    # Logging
    parser.add_argument('--save_dir', type=str, default='saved_models')
    parser.add_argument('--task_name', type=str, default='lm')

    args = parser.parse_args()

    return args


def train_epoch(model, batches, optimizer, criterion, params):
    model.train()
    epoch_loss = 0.
    for batch in batches:
        # Forward
        optimizer.zero_grad()
        sents = pad_sequence(batch)

        output = model(sents[:-1])
        vocab_size = output.size(-1)
        loss = criterion(output.view(-1, vocab_size), sents[1:].view(-1))
        loss.backward()

        # Gradient clip
        nn.utils.clip_grad_norm_(model.parameters(), params.clip)

        epoch_loss += loss.item()
        optimizer.step()

    return epoch_loss / len(batches)


def eval_samples(model, batches, criterion, params):
    model.eval()
    total_loss = 0.
    for batch in batches:
        sents = pad_sequence(batch)

        output = model(sents[:-1])
        vocab_size = output.size(-1)
        loss = criterion(output.view(-1, vocab_size), sents[1:].view(-1))

        total_loss += loss.item()

    return total_loss / len(batches)
    

def train(model, data, params):
    """Trains a language model on a corpus."""

    log = Logger(20, params.task_name)

    train_batches = get_seq_batches(data.train_data, params.batch_size, params.primal)
    valid_batches = get_seq_batches(data.valid_data, params.batch_size, params.primal)

    log.info('Number of steps per epoch: {}'.format(len(train_batches)))
    print('Number of steps per epoch: {}'.format(len(train_batches)))
    log.info('Batch size: {}'.format(params.batch_size))
    print('Batch size: {}'.format(params.batch_size))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    best_valid_loss = None
    # Loop over epochs.
    for epoch in range(params.epochs):
        log.info('Epoch: {}'.format(epoch))
        print('Epoch: {}'.format(epoch))

        train_loss = train_epoch(
            model, train_batches, optimizer, criterion, params)
        log.info('Training loss: {:.3f}'.format(train_loss))
        print('Training loss: {:.3f}'.format(train_loss))

        valid_loss = eval_samples(model, valid_batches, criterion, params)
        log.info('Validation loss: {:.3f} | ppl: {:.3f}'.format(
            valid_loss, math.exp(valid_loss)))
        print('Validation loss: {:.3f} | ppl: {:.3f}'.format(
            valid_loss, math.exp(valid_loss)))

        if not best_valid_loss or valid_loss < best_valid_loss:
            model.save(os.path.join(params.save_dir, '{}.pt'.format(params.task_name)))
            best_valid_loss = valid_loss

    log.info('Finished training!')
    print('Finished training!')


def evaluate(model, data, params, split):
    """Evaluates a language model on a corpus.

    Args:
        model (LanguageModel)
        data (Corpus)
        params (namespace)
        split (str): Split to evaluate (valid, dev or test).
    """
    model.load(os.path.join(params.save_dir, '{}.pt'.format(params.task_name)))
    data_split = eval('data.{}_data'.format(split))
    batches = get_seq_batches(data_split, params.batch_size, params.primal)
    criterion = nn.NLLLoss()
    
    eval_loss = eval_samples(model, batches, criterion, params)
    print('Evaluation loss: {:.3f} | ppl: {:.3f}'.format(
        eval_loss, math.exp(eval_loss)))


if __name__ == '__main__':
    params = get_params()

    data = Corpus(params)
    vocab = data.utter_vocab if params.primal else data.query_vocab
    model = LanguageModel(
        len(vocab),
        params.emb_dim,
        params.hidden_dim,
        params.num_layers,
        dropout=params.dropout,
        tie_weights=params.tie_weights).to(device)

    print('=====================Model Parameters=====================')
    for name, param in model.named_parameters():
        print(name, param.requires_grad, param.is_cuda, param.size())
        # assert param.is_cuda

    sys.stdout.flush()

    if params.train:
        train(model, data, params)
    if params.evaluate and 'valid' in params.evaluate_split:
        evaluate(model, data, params, split='valid')
    if params.evaluate and 'dev' in params.evaluate_split:
        evaluate(model, data, params, split='dev')
    if params.evaluate and 'test' in params.evaluate_split:
        evaluate(model, data, params, split='test')
