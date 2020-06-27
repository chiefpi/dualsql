import torch

from data_utils.vocabulary import UNK_TOK, EOS_TOK
from model_utils.tensor import lens2mask

class RewardModel:
    
    def __init__(self, utter_lm, query_lm, utter_vocab, query_vocab, sp_device='cpu', qg_device='cpu'):
        self.utter_lm = utter_lm.to(sp_device) # utterance language model
        self.query_lm = query_lm.to(qg_device) # query language model
        self.utter_vocab = utter_vocab
        self.query_vocab = query_vocab
        self.sp_device = sp_device
        self.qg_device = qg_device

    def forward(self, *args, choice='sp_val'):
        if choice == 'sp_val':
            return self.sp_validity_reward(*args)
        elif choice == 'qg_val':
            return self.qg_validity_reward(*args)
        elif 'rec' in choice:
            return self.reconstruction_reward(*args)
        else:
            raise ValueError('Unknown reward choice')

    def sp_validity_reward(self, query):
        """Calculates query language model length normalized log probability."""
        lens = [len(each) for each in input_idxs]
        max_len = max(lens)
        input_idxs = [sent + [self.vocab.lf2id[PAD]] * (max_len - len(sent)) for sent in input_idxs]
        input_tensor = torch.tensor(input_idxs, dtype=torch.long, device=self.qg_device)
        lens = torch.tensor(lens, dtype=torch.long, device=self.qg_device)
        self.query_lm.eval()
        with torch.no_grad():
            log_prob = self.query_lm.sentence_log_prob(input_tensor, lens).cpu()
        return log_prob

    def qg_validity_reward(self, utter):
        """Calculates utterance language model length normalized log probability."""
        lens = [len(each) for each in utter]
        max_len = max(lens)
        input_idxs = [sent + [self.vocab.word2id[PAD]] * (max_len - len(sent)) for sent in input_idxs]
        input_tensor = torch.tensor(input_idxs, dtype=torch.long, device=self.sp_device)
        lens = torch.tensor(lens, dtype=torch.long, device=self.sp_device)
        self.utter_lm.eval()
        with torch.no_grad():
            log_prob = self.utter_lm.sentence_log_prob(input_tensor, lens).cpu()
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

    def __call__(self, *args, **kargs):
        return self.forward(*args, **kargs)
