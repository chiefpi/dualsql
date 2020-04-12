import time
import os
import sys
import argparse

import torch
import torch.nn as nn

from data_utils.data import Corpus
from model.dualsql import DualSQL


def interpret_args():
    parser = argparse.ArgumentParser()


    return parser.parse_args()


def train(model, data, params):
    pass


def evaluate(model, data, params, last_save_file, split):
    # Turn on evaluation mode which disables dropout.
    pass

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
    model = DualSQL(
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

    # # Load the best saved model.
    # with open(params.save, 'rb') as f:
    #     model = torch.load(f)
    #     # Make rnn params a continuous chunk to speed up forward pass.
    #     if params.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
    #         model.rnn.flatten_parameters()

    # # Run on test data.
    # test_loss = evaluate(test_data)
    # print('=' * 89)
    # print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    #     test_loss, math.exp(test_loss)))


if __name__ == "__main__":
    main()