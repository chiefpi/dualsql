import torch

from torch.nn.utils.rnn import pad_sequence

from data_utils.vocabulary import UNK_TOK, EOS_TOK
from model_utils.tensor import lens2mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RewardModel:
    
    def __init__(self, utter_lm, query_lm, utter_vocab, query_vocab):
        self.utter_lm = utter_lm.to(device) # utterance language model
        self.query_lm = query_lm.to(device) # query language model
        self.utter_vocab = utter_vocab
        self.query_vocab = query_vocab

    def validity_reward(self, seqs, primal):
        """Calculates query language model length normalized log probability."""
        lens = [seq.size(0) for seq in seqs]
        seqs = pad_sequence(seqs).to(device)
        seqs_tensor = torch.LongTensor(seqs).to(device)
        lens = torch.LongTensor(lens).to(device)

        lm = self.query_lm if primal else self.utter_lm
        lm.eval()
        with torch.no_grad():
            log_prob = lm.sentence_log_prob(seqs_tensor, lens)

        return log_prob

    def reconstruction_reward(self, log_scores, references, lens):
        """Calculates log-likelihood.
        Args:
            log_scores: bsize x max_out_len x vocab_size[ + MAX_OOV_NUM]
            references: bsize x max_out_len
            lens: len for each sample
        Returns:
            bsize x 
        """
        mask = lens2mask(lens)
        pick_score = torch.gather(log_scores, dim=-1, index=references.unsqueeze(-1)).squeeze(-1)
        masked_score = mask.float() * pick_score
        reward = masked_score.sum(dim=1)

        return reward
