import time
import math
import os
import argparse

import torch
import torch.nn as nn

from data_utils.data import Corpus
from model.language_model import LanguageModel


def interpret_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/sparc',
                        help='location of the data data')
    # Model
    parser.add_argument('--emb_dim', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--hidden_dim', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers')
    # Training
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40, 
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    # parser.add_argument('--bptt', type=int, default=35,
    #                     help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    # parser.add_argument('--tied', action='store_true',
    #                     help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--evaluate_split', type=str, default='dev',
                        help='')
    # Log
    parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='saved_models/sparc/language_model/model.pt',
                        help='path to save the final model')

    return parser.parse_args()


def train(model, data, params):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    vocab_size = len(data.dictionary)
    hidden = model.init_hidden(params.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, params.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), params.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % params.log_interval == 0 and batch > 0:
            cur_loss = total_loss / params.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // params.bptt, lr,
                elapsed * 1000 / params.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        # Loop over epochs.
        
    lr = params.lr
    best_val_loss = None

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, params.epochs+1):
            epoch_start_time = time.time()
            train()
            val_loss = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(params.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


def evaluate(model, data, params, last_save_file, split):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    vocab_size = len(data.dictionary)
    if params.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, params.bptt):
            data, targets = get_batch(data_source, i)
            if params.model == 'Transformer':
                output = model(data)
                output = output.view(-1, vocab_size)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def main():
    """Main function that trains and/or evaluates a language model."""
    params = interpret_args()

    # Set the random seed manually for reproducibility
    torch.manual_seed(params.seed)
    device = torch.device("cuda" if params.cuda else "cpu")

    # Prepare the dataset into the proper form
    data = Corpus(params.data)

    # Build the model
    vocab_size = len(data.dictionary)
    model = LanguageModel(
        vocab_size,
        params.emb_dim,
        params.hidden_dim,
        params.num_layers,
        params.dropout,
        params.tied).to(device)

    # model = LanguageModel(
    #     params,
    #     data.input_vocabulary,
    #     data.output_vocabulary,
    #     data.output_vocabulary_schema,
    #     data.anonymizer if params.anonymize and params.anonymization_scoring else None)

    model = model.cuda()
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

    # Load the best saved model.
    with open(params.save, 'rb') as f:
        model = torch.load(f)
        # Make rnn params a continuous chunk to speed up forward pass.
        if params.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
            model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))


if __name__ == "__main__":
    main()