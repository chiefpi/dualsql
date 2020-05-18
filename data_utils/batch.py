import copy
import math
import random

import torch


def get_all_turns(
        dataset,
        max_utter_len=math.inf,
        max_query_len=math.inf):
    """Gets all turns in a dataset.
    
    Returns:
        list of Turn
    """

    return [turn for interaction in dataset.interactions
        for turn in interaction.turns
        if turn.len_valid(max_utter_len, max_query_len)]

def get_all_interactions(
        dataset,
        max_inter_len=math.inf,
        max_utter_len=math.inf,
        max_query_len=math.inf,
        sorted_by_len=False):
    """Gets all interactions in a dataset that fit the criteria.

    Args:
        dataset (DatasetSplit): The dataset to use.
        max_inter_len (int): Maximum interaction len to keep.
        max_utter_len (int): Maximum utter sequence len to keep.
        max_query_len (int): Maximum query sequence len to keep.
        sorted_by_len (bool): Whether to sort the interactions by interaction len.

    Returns:
        list of Interaction
    """
    interactions = [interaction for interaction in dataset.interactions
        if len(interaction) <= max_inter_len and
        interaction.length_valid(max_utter_len, max_query_len)]

    return sorted(interactions, key=len, reverse=True) if sorted_by_len else interactions

def seq2tensor(seqs):
    lens = [len(seq) for seq in seqs]
    lens_tensor = torch.tensor(lens, dtype=torch.long)

    max_len = max(lens)
    pad_seqs = [seq + [0] * (max_len - len(seq)) for seq in seqs]
    seqs_tensor = torch.tensor(pad_seqs, dtype=torch.long)

    return seqs_tensor, lens_tensor

def get_seq_batches(
        dataset,
        batch_size,
        primal,
        max_len=math.inf,
        randomize=True):
    """
    Returns:
        list of list of torch.tensor
    """
    seqs = dataset.get_all_utterances() if primal else dataset.get_all_queries()
    if randomize:
        random.shuffle(seqs)

    seqs = [torch.LongTensor(seq) for seq in seqs]

    return [seqs[i:i+batch_size]
        for i in range(0, len(seqs), batch_size)]

# def fix_parentheses(seq):
#     """Balances parentheses for a sequence.

#     Args:
#         seq (list of str): Contains an EOS token.
#     """
#     return seq[:-1] + list(")" * (seq.count("(")-seq.count(")"))) + [seq[-1]]


# def flatten_sequence(seq):
#     """Removes EOS token from a sequence."""
    
#     return fix_parentheses(seq[:-1] if seq[-1] == EOS_TOK else seq)
