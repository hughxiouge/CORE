from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset

class E2T_TrainDataset(Dataset):
    def __init__(self, pairs, nentity, ntype, negative_sample_size, mode):
        self.len = len(pairs)
        self.pairs = pairs
        self.pair_set = set(pairs)
        self.nentity = nentity
        self.ntype = ntype
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.entity_count, self.type_count = self.count_frequency(pairs)
        self.true_type = self.get_true_type(pairs)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.pairs[idx]

        entity, ent_type = positive_sample

        subsampling_weight = self.entity_count[entity] + self.type_count[ent_type]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.ntype, size=self.negative_sample_size*2)
            mask = np.in1d(
                negative_sample, 
                self.true_type[entity], 
                assume_unique=True, 
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.LongTensor(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)
            
        return positive_sample, negative_sample, subsampling_weight, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
    @staticmethod
    def count_frequency(pairs, start=4):
        '''
        Get frequency of entity, type
        The frequency will be used for subsampling like word2vec
        '''
        entity_count = {}
        type_count = {}
        for entity, ent_type in pairs:
            if entity not in entity_count:
                entity_count[entity] = start
            else:
                entity_count[entity] += 1
            
            if ent_type not in type_count:
                type_count[ent_type] = start
            else:
                type_count[ent_type] += 1

        return entity_count, type_count
    
    @staticmethod
    def get_true_type(pairs):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        true_type = {}
        for entity, ent_type in pairs:
            if entity not in true_type:
                true_type[entity] = []
            true_type[entity].append(ent_type)

        for entity in true_type:
            true_type[entity] = np.array(list(set(true_type[entity])))

        return true_type

    
class E2T_TestDataset(Dataset):
    def __init__(self, pairs, all_true_pairs, nentity, ntype, mode):
        self.len = len(pairs)
        self.pair_set = set(all_true_pairs)
        self.pairs = pairs
        self.nentity = nentity
        self.ntype = ntype
        self.mode = mode

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        entity, ent_type = self.pairs[idx]
        # create filter
        tmp = [(0, rand_type) if (entity, rand_type) not in self.pair_set
                   else (-100000, rand_type) for rand_type in range(self.ntype)]
        tmp[ent_type] = (0, ent_type)
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((entity, ent_type))
            
        return positive_sample, negative_sample, filter_bias, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode

class OneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        data = next(self.iterator)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data