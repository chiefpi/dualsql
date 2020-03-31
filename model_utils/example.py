import sys
import os
import numpy as np

from domain import Domain

def split_dataset(dataset, split_ratio=1.0):
    assert split_ratio >= 0. and split_ratio <= 1.0
    index = np.arange(len(dataset))
    np.random.shuffle(index)
    splt = int(len(dataset) * split_ratio)
    first = [dataset[idx] for idx in index[:splt]]
    second = [dataset[idx] for idx in index[splt:]]

    return first, second


class Example():

    __slots__ = ('question', 'logical_form', "conf")

    def __init__(self, question='', logical_form='', conf=1.0):
        """
        @question: string of question
        @logical_form: string of logical form
        @conf: 
        """
        super(Example, self).__init__()
        # tokenizing
        self.question = [each for each in question.split(' ') if each != '']
        self.logical_form = [each for each in logical_form.split(' ') if each != '']
        # entity mapping
        # self.mapped_question = Example.db.entity_mapping(self.question)
        # self.mapped_logical_form = Example.db.reverse_entity_mapping(self.logical_form, self.question)
        self.conf = conf

    @classmethod
    def set_domain(cls, dataset):
        cls.dataset = dataset # dataset name
        # cls.db = Lexicon(dataset)
        cls.domain = Domain.from_dataset(dataset) # class Domain object
        if dataset in ['geo', 'atis']:
            cls.file_paths = [
                os.path.join('data', dataset, dataset + '_train.tsv'),
                os.path.join('data', dataset, dataset + '_dev.tsv'),
                os.path.join('data', dataset, dataset + '_test.tsv')
            ]
            cls.extra_path = os.path.join('data', dataset, dataset + '_extra.tsv')
        else: #Overnight
            cls.file_paths = [
                os.path.join('data', 'overnight', dataset + '_train.tsv'),
                os.path.join('data', 'overnight', dataset + '_test.tsv')
            ]
            cls.extra_path = os.path.join('data', 'overnight', dataset + '_extra.tsv')

    @classmethod
    def load_dataset(cls, choice='train'):
        """
            return example list of train, test or extra
        """
        if choice == 'train':
            if len(cls.file_paths) == 2:
                # no dev dataset, split train dataset
                train_dataset = cls.load_dataset_from_file(cls.file_paths[0])
                train_dataset, dev_dataset = split_dataset(train_dataset, split_ratio=0.8)
            else:
                assert len(cls.file_paths) == 3
                train_dataset = cls.load_dataset_from_file(cls.file_paths[0])
                dev_dataset = cls.load_dataset_from_file(cls.file_paths[1])
            return train_dataset, dev_dataset
        elif choice == 'test':
            test_dataset = cls.load_dataset_from_file(cls.file_paths[-1])
            return test_dataset
        else:
            extra_dataset = cls.load_dataset_from_file(cls.extra_path)
            return extra_dataset

    @classmethod
    def load_dataset_from_file(cls, path):
        ex_list = []
        with open(path, 'r') as infile:
            for line in infile:
                line = line.strip()
                if line == '': continue
                q, lf = line.split('\t')
                ex_list.append(cls(q.strip(), lf.strip()))
        return ex_list
