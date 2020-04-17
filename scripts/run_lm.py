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

from logger import Logger
from data_utils import corpus
from data_utils.corpus import Corpus
from model.language_model import LanguageModel
from model_utils.train import evaluate_turn_sample, \
    train_epoch_with_turns, evaluate_using_predicted_queries


# Set the random seed manually for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_params():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_dir', type=str,
        default='data/sparc')
    parser.add_argument('--raw_train_filename', type=str,
        default='data/sparc_data_removefrom/train.pkl')
    parser.add_argument('--raw_validation_filename', type=str,
        default='data/sparc_data_removefrom/dev.pkl')
    # TODO: redundant params of data
    # Model
    parser.add_argument('--task', choices=['utterance', 'query'])
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--tied', action='store_true')
    # Training
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--lr', type=float, default=20)
    parser.add_argument('--clip', type=float, default=0.25)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=20,
        help='tie the word embedding and softmax weights')
    parser.add_argument('--evaluate_split', choices=['valid', 'dev', 'test'])
    # Logging
    parser.add_argument('--log_dir', type=str, 
        default='logs/language_model')
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


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(batch, i):
    pass


def train(model, data, params):
    """Trains a language model on a corpus."""

    log = Logger(os.path.join(params.log_dir, params.log_file), 'w')
    num_train = corpus.num_turns(data.train_data)
    log.put('Total number of training turns: {:d}' % num_train)

    train_batches = data.get_turn_batches(params.batch_size)
    # evaluation samples
    train_samples = data.get_random_turns(params.train_evaluation_size)
    valid_examples = data.get_all_turns(data.valid_data)

    log.put('Number of steps per epoch: {:d}' % len(train_batches))
    log.put('Batch size: {:d}' % params.batch_size)

    lr = params.lr
    criterion = nn.NLLLoss()
    best_valid_loss = None
    # Loop over epochs.
    for epoch in range(params.epochs):
        log.put('Epoch: {:d}' % epoch)
        model.train()
        epoch_loss = 0.
        hidden = model.init_hidden(params.batch_size)
        for batch in train_batches:
            # Detach the hidden state from how it was previously produced.
            model.zero_grad()
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
            loss = criterion(output, targets)
            loss.backward()

            # Gradient clip
            nn.utils.clip_grad_norm_(model.parameters(), params.clip)
            for p in model.parameters():
                p.data.add_(-lr, p.grad.data)

            epoch_loss += loss.item()

        log.put('Train epoch loss: {:.3f}'.format(epoch_loss))

        train_eval_loss = evaluate_turn_sample(train_samples)
        log.put('Train evaluation loss: {:.3f} | ppl: {:.3f}'.format(
            train_eval_loss, math.exp(train_eval_loss)))

        valid_loss = evaluate_turn_sample(valid_examples)
        log.put('Validation loss: {:.3f} | ppl: {:.3f}'.format(
            valid_loss, math.exp(valid_loss)))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or valid_loss < best_valid_loss:
            with open(os.path.join(params.log_dir, params.save_file), 'wb') as f:
                torch.save(model, f)
            best_val_loss = valid_loss
        else:
            lr /= 4.0

    log.put('Finished training!')
    log.close()


def evaluate(model, data, params, split):
    """Evaluates a language model on a corpus.

    Args:
        model (LanguageModel)
        data (Corpus)
        params (namespace)
        split (str): Split to evaluate (valid, dev or test).
    """
    model.load(os.path.join(params.log_dir, params.language_model_file))
    data_split = eval('data.%s_data' % split)
    examples = data.get_all_turns(data_split)

    # Turn on evaluation mode which disables dropout.
    model.eval()
    epoch_loss = 0.
    hidden = model.init_hidden(params.batch_size)
    criterion = nn.NLLLoss()
    with torch.no_grad():
        for i in range(0, data.size(0) - 1, params.bptt):
            data, targets = get_batch(data, i)
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            epoch_loss += len(data) * criterion(output, targets).item()
    test_loss = epoch_loss / (len(data) - 1)
    print('=' * 89)
    print('est loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))


def main():
    params = get_params()

    data = Corpus(params)
    model = LanguageModel(
        vocab_size=len(data.input_vocabulary),
        emb_file=params.emb_file,
        hidden_dim=params.hidden_dim,
        num_layers=params.num_layers,
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


if __name__ == '__main__':
    main()
