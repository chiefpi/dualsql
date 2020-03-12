import argparse

def optparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, help='pseudo method for semantic parsing')
    parser.add_argument('--testing', action='store_true', help='Only test your model (default is training && testing)')
    parser.add_argument('--dataset', required=True, help='which dataset to experiment on')
    parser.add_argument('--read_model_path', help='Testing mode, load sp and qg model path')
    # model params
    parser.add_argument('--read_sp_model_path', required=True, help='pretrained sp model')
    parser.add_argument('--read_qg_model_path', required=True, help='pretrained qg model path')
    parser.add_argument('--read_qlm_path', required=True, help='language model for natural language questions')
    parser.add_argument('--read_lflm_path', required=True, help='language model for logical form')
    # pseudo training paras
    parser.add_argument('--reduction', choices=['sum', 'mean'], default='sum')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-5, help='weight decay (L2 penalty)')
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--test_batchSize', type=int, default=128, help='input batch size in decoding')
    parser.add_argument('--max_norm', type=float, default=5, help="threshold of gradient clipping (2-norm)")
    parser.add_argument('--max_epoch', type=int, default=100, help='max number of epochs to train for')
    # special paras
    parser.add_argument('--sample', type=int, default=5, help='size of sampling during training in dual learning')
    parser.add_argument('--beam', default=5, type=int, help='used during decoding time')
    parser.add_argument('--n_best', default=1, type=int, help='used during decoding time')
    parser.add_argument('--alpha', type=float, default=0.5, help='coefficient which combines sp valid and reconstruction reward')
    parser.add_argument('--beta', type=float, default=0.5, help='coefficient which combines qg valid and reconstruction reward')
    parser.add_argument('--cycle', choices=['sp', 'qg', 'sp+qg'], default='sp+qg', help='whether use cycle starts from sp/qg')
    parser.add_argument('--labeled', type=float, default=1.0, help='ratio of labeled samples')
    parser.add_argument('--unlabeled', type=float, default=1.0, help='ratio of unlabeled samples')
    parser.add_argument('--deviceId', type=int, nargs=2, default=[-1, -1], help='device for semantic parsing and question generation model respectively')
    parser.add_argument('--seed', type=int, default=999, help='set initial random seed')
    parser.add_argument('--extra', action='store_true', help='whether use synthesized logical forms')
    opt = parser.parse_args(args)

    # Some Arguments Check
    assert opt.labeled > 0.
    assert opt.unlabeled >= 0. and opt.unlabeled <= 1.0
    return opt