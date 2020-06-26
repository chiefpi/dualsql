"""Pretrains the model with labeled data."""

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
from data_utils.batch import get_all_interactions
from data_utils.corpus import Corpus
from models.dualsql import DualSQL


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
    parser.add_argument('--use_bert', action='store_true')
    parser.add_argument('--use_editing', action='store_true')
    parser.add_argument('--max_gen_len', type=int, default=1000)
    parser.add_argument('--max_turns_to_keep', type=int, default=5)
    # Training
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--evaluate_split', choices=['valid', 'dev', 'test'])
    # Logging
    parser.add_argument('--save_dir', type=str, default='saved_models')
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--task_name', type=str, default='sp')

    args = parser.parse_args()

    return args


def train_epoch(model, interactions, optimizer, criterion, params):
    torch.autograd.set_detect_anomaly(True)
    model.train()
    epoch_loss = 0.
    for interaction in interactions:
        # Forward
        optimizer.zero_grad()

        dists_seq = model(interaction, params.primal, params.max_gen_len, params.force)
        tgt_seqs = interaction.query_seqs() if params.primal else interaction.utter_seqs()
        losses = []
        for dists, tgt in zip(dists_seq, tgt_seqs):
            print(dists.size(), tgt)
            losses.append(criterion(dists.squeeze(), torch.LongTensor(tgt, device=device)))
        loss = torch.sum(torch.stack(losses))
        loss.backward()

        epoch_loss += loss.item()
        optimizer.step()

    return epoch_loss / len(interactions)


def eval_samples(model, interactions, criterion, params):
    model.eval()
    total_loss = 0.
    for interaction in interactions:
        scores_seqs = model(interaction, params.primal, params.max_gen_len, params.force)
        gt_seqs = interaction.query_seqs if params.primal else interaction.utter_seqs
        for scores, gt in scores_seqs, gt_seqs:
            loss = criterion(scores, gt)
        loss.backward()

        epoch_loss += loss.item()

    return total_loss / len(interactions)


def train(model, data, params):
    log = Logger(20, params.task_name)

    train_interactions = get_all_interactions(data.train_data)
    valid_interactions = get_all_interactions(data.valid_data)

    log.info('Number of steps per epoch: {}'.format(len(train_interactions)))
    print('Number of steps per epoch: {}'.format(len(train_interactions)))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    best_valid_loss = None
    # Loop over epochs.
    for epoch in range(params.epochs):
        log.info('Epoch: {}'.format(epoch))
        print('Epoch: {}'.format(epoch))

        train_loss = train_epoch(model, train_interactions, optimizer, criterion, params)
        log.info('Training loss: {:.3f}'.format(train_loss))
        print('Training loss: {:.3f}'.format(train_loss))

        valid_loss = eval_samples(model, valid_interactions, criterion, params)
        log.info('Validation loss: {:.3f}'.format(valid_loss))
        print('Validation loss: {:.3f}'.format(valid_loss))

        if not best_valid_loss or valid_loss < best_valid_loss:
            model.save(os.path.join(params.save_dir, '{}.pt'.format(params.task_name)))
            best_valid_loss = valid_loss

    log.info('Finished training!')
    print('Finished training!')


def evaluate(model, data, params, split):
    model.load(os.path.join(params.save_dir, '{}.pt'.format(params.task_name)))
    data_split = eval('data.{}_data'.format(split))
    interactions = get_all_interactions(data_split)
    criterion = nn.NLLLoss()

    eval_loss = eval_samples(model, interactions, criterion, params)
    print('Evaluation loss: {:.3f}'.format(eval_loss))


if __name__ == "__main__":
    params = get_params()

    data = Corpus(params)
    vocab = data.utter_vocab if params.primal else data.query_vocab
    model = DualSQL(
        data.utter_vocab,
        data.query_vocab,
        data.schema_vocab,
        # params.embedding_filename,
        freeze=params.freeze,
        dropout=params.dropout,
        use_editing=params.use_editing,
        max_turns_to_keep=params.max_turns_to_keep,
        use_bert=params.use_bert).to(device)

    if params.pretrained:
        model.load()

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