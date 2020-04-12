import os
import sys
import argparse

from logger import Logger
from model.dualsql import DualSQL

def get_params():
    parser = argparse.ArgumentParser()
    # Locations
    parser.add_argument('--embedding_filename', type=str)
    parser.add_argument('--input_vocabulary_filename', type=str, default='input_vocabulary.pkl')
    parser.add_argument('--output_vocabulary_filename', type=str, default='output_vocabulary.pkl')
    # Embedder
    parser.add_argument('--use_bert', action='store_true')
    parser.add_argument('--input_embedding_size', type=int, default=300)
    parser.add_argument('--output_embedding_size', type=int, default=300)
    # Encoder
    parser.add_argument('--schema_encoder_input_size', type=int, default=300)
    parser.add_argument('--schema_encoder_state_size', type=int, default=300)
    parser.add_argument('--schema_attention_key_size', type=int, default=300)
    parser.add_argument('--use_text_schema_attention', action='store_true')
    parser.add_argument('--encoder_state_size', type=int, default=300)
    # Decoder
    parser.add_argument('--use_editing_mechanism', action='store_true')
    # Training
    parser.add_argument('--train', action='store_true')
    # Logs
    parser.add_argument('--log_dir', type=str, default='logs')

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


def train(model, data, params):


def main():



if __name__ == "__main__":
    main()