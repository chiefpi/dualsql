import copy
import math

from data_utils.vocab import EOS_TOK


def fix_parentheses(seq):
    """Balances parentheses for a sequence.

    Args:
        seq (list of str): Contains an EOS token.
    """
    return seq[:-1] + list(")" * (seq.count("(")-seq.count(")"))) + [seq[-1]]


def flatten_sequence(seq):
    """Removes EOS token from a sequence."""
    
    return fix_parentheses(seq[:-1] if seq[-1] == EOS_TOK else seq)


def get_minibatch_sp(ex_list, vocab, device, copy=False):
    inputs = [ex.question for ex in ex_list]
    lens = [len(ex) for ex in inputs]
    lens_tensor = torch.tensor(lens, dtype=torch.long, device=device)

    max_len = max(lens)
    padded_inputs = [sent + [PAD] * (max_len - len(sent)) for sent in inputs]
    inputs_idx = [[vocab.word2id[w] if w in vocab.word2id else vocab.word2id[UNK] for w in sent] for sent in padded_inputs]
    inputs_tensor = torch.tensor(inputs_idx, dtype=torch.long, device=device)

    outputs = [ex.logical_form for ex in ex_list]
    bos_eos_outputs = [[BOS] + sent + [EOS] for sent in outputs]
    out_lens = [len(each) for each in bos_eos_outputs]
    max_out_len = max(out_lens)
    padded_outputs = [sent + [PAD] * (max_out_len - len(sent)) for sent in bos_eos_outputs]
    outputs_idx = [[vocab.lf2id[w] if w in vocab.lf2id else vocab.lf2id[UNK] for w in sent] for sent in padded_outputs]
    outputs_tensor = torch.tensor(outputs_idx, dtype=torch.long, device=device)
    out_lens_tensor = torch.tensor(out_lens, dtype=torch.long, device=device)

    if copy: # pointer network need additional information
        mapped_inputs = [ex.mapped_question for ex in ex_list]
        oov_list, copy_inputs = [], []
        for sent in mapped_inputs:
            tmp_oov_list, tmp_copy_inputs = [], []
            for idx, word in enumerate(sent):
                if word not in vocab.lf2id and word not in tmp_oov_list and len(tmp_oov_list) < MAX_OOV_NUM:
                    tmp_oov_list.append(word)
                tmp_copy_inputs.append(
                    (
                        vocab.lf2id.get(word, vocab.lf2id[UNK]) if word in vocab.lf2id or word not in tmp_oov_list \
                        else len(vocab.lf2id) + tmp_oov_list.index(word) # tgt_vocab_size + oov_id
                    )
                )
            tmp_oov_list += [UNK] * (MAX_OOV_NUM - len(tmp_oov_list))
            oov_list.append(tmp_oov_list)
            copy_inputs.append(tmp_copy_inputs)

        copy_tokens = [
            torch.cat([
                torch.zeros(len(each), len(vocab.lf2id) + MAX_OOV_NUM, dtype=torch.float)\
                    .scatter_(-1, torch.tensor(each, dtype=torch.long).unsqueeze(-1), 1.0),
                torch.zeros(max_len - len(each), len(vocab.lf2id) + MAX_OOV_NUM, dtype=torch.float)
            ], dim=0)
            for each in copy_inputs
        ]
        copy_tokens = torch.stack(copy_tokens, dim=0).to(device) # bsize x src_len x (tgt_vocab + MAX_OOV_NUM)

        dec_outputs = [
            [
                len(vocab.lf2id) + oov_list[idx].index(tok)
                    if tok not in vocab.lf2id and tok in oov_list[idx] \
                    else vocab.lf2id.get(tok, vocab.lf2id[UNK])
                for tok in sent
            ] + [vocab.lf2id[PAD]] * (max_out_len - len(sent))
            for idx, sent in enumerate(bos_eos_outputs)
        ]
        dec_outputs_tensor = torch.tensor(dec_outputs, dtype=torch.long, device=device)
    else:
        dec_outputs_tensor, copy_tokens, oov_list = outputs_tensor, None, []

    return inputs_tensor, lens_tensor, outputs_tensor, dec_outputs_tensor, out_lens_tensor, copy_tokens, oov_list, (inputs, outputs)

