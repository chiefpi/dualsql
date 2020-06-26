import torch
from models.dualsql import DualSQL
from data_utils.vocab import Vocab
from data_utils.interaction import Interaction
from data_utils.turn import Turn
from data_utils.schema import Schema

uvocab = Vocab([['this', 'is', 'a', 'test']], data_type='utter')
qvocab = Vocab([['dont', 'care', 'test']], data_type='query')
svocab = Vocab([['just', 'a', 'test', '.']], data_type='schema')
print(len(uvocab), len(qvocab), len(svocab))

model = DualSQL(
    uvocab,
    qvocab,
    svocab,
    freeze=True,
    dropout=0,
    use_editing=False,
    use_bert=False)

ex = {'utterance': 'this is a test', 'sql': ['dont', 'test']}
it = Interaction([Turn(ex) for i in range(5)], Schema([['just', '.', 'test'], ['a', '.', 'test'], ['just']], ['just.test', 'a.test', 'just']), 'int0')
it.str2index(svocab, uvocab, qvocab)
dist_seqs = model(it, False, 20, True)
for dist in dist_seqs:
    print(dist.size())
dist_seqs = model(it, True, 20, True)
for dist in dist_seqs:
    print(dist.size())

