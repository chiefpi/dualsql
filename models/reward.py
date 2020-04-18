import torch

from data_utils.vocabulary import UNK_TOK, EOS_TOK
from model_utils.tensor import lens2mask

# TODO
class RewardModel:

    def __init__(self, utter_lm, query_lm, vocab,
            sp_device='cpu', qg_device='cpu'):
        super(RewardModel, self).__init__()
        self.utter_lm = utter_lm.to(sp_device) # utterance language model
        self.query_lm = query_lm.to(qg_device) # query language model
        self.vocab = vocab
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

    def sp_validity_reward(self, queries):
        # calculate query language model length normalized log probability
        input_idxs = [[self.vocab.lf2id[BOS]] + [self.vocab.lf2id[word] if word in self.vocab.lf2id else self.vocab.lf2id[UNK] for word in sent] + [self.vocab.word2id[EOS]] for sent in queries]
        lens = [len(each) for each in input_idxs]
        max_len = max(lens)
        input_idxs = [sent + [self.vocab.lf2id[PAD]] * (max_len - len(sent)) for sent in input_idxs]
        input_tensor = torch.tensor(input_idxs, dtype=torch.long, device=self.qg_device)
        lens = torch.tensor(lens, dtype=torch.long, device=self.qg_device)
        self.query_lm.eval()
        with torch.no_grad():
            log_prob = self.query_lm.sentence_log_prob(input_tensor, lens).cpu()
        # grammar check
        # TODO: find sql validator
        domain = Example.domain
        ans = domain.is_valid(domain.obtain_denotations(domain.normalize(queries)))
        grammar = torch.tensor(ans, dtype=torch.float, requires_grad=False)
        val_reward = 0.5 * log_prob + 0.5 * grammar
        return val_reward

    def qg_validity_reward(self, utterances):
        # calculate utterance language model length normalized log probability
        input_idxs = [[self.vocab.word2id[BOS]] + [self.vocab.word2id[word] if word in self.vocab.word2id else self.vocab.word2id[UNK] for word in sent] + [self.vocab.word2id[EOS]] for sent in utterances]
        lens = [len(each) for each in input_idxs]
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
        pick_score = torch.gather(log_scores, dim=-1, index=references.unsqueeze(dim=-1)).squeeze(dim=-1)
        masked_score = mask.float() * pick_score
        reward = masked_score.sum(dim=1)
        return reward

    def __call__(self, *args, **kargs):
        return self.forward(*args, **kargs)
