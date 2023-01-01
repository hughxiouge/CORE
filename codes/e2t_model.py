from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter

from torch.utils.data import DataLoader

from e2t_dataloader import E2T_TestDataset

class E2TModel(nn.Module):
    def __init__(self, nentity, ntype, entity_dim, type_dim, gamma,
                 entity_parameter, id2type, id2entity,
                 double_entity_embedding=False, double_type_embedding=False):
        super(E2TModel, self).__init__()
        self.nentity = nentity
        self.ntype = ntype
        self.entity_dim = entity_dim
        self.type_dim = type_dim
        self.epsilon = 2.0
        self.id2type = id2type
        self.id2entity = id2entity
        self.e2t_rank_prediction = []

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / type_dim]), 
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(data=entity_parameter, requires_grad=False)
        
        # self.type_embedding = nn.Parameter(torch.zeros(ntype, self.type_dim))

        if double_type_embedding:
            self.type_embedding = nn.Parameter(torch.zeros(ntype, self.type_dim*2))
        else:
            self.type_embedding = nn.Parameter(torch.zeros(ntype, self.type_dim))
        
        nn.init.uniform_(
            tensor=self.type_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
    
        if double_entity_embedding and double_type_embedding:
            self.M = nn.Parameter(torch.zeros(2*self.entity_dim, 2*self.type_dim))
        else:
            self.M = nn.Parameter(torch.zeros(self.entity_dim, self.type_dim))
        nn.init.uniform_(
            tensor=self.M, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
    def forward(self, sample, mode='single'):
        if mode == 'single':
            entity_feature = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            type_feature = torch.index_select(
                self.type_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)

        elif mode == 'choice_batch':
            
            entity_part, type_part = sample
            batch_size, negative_sample_size = type_part.size(0), type_part.size(1)            
            entity_feature = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=entity_part[:, 0]
            ).unsqueeze(1)
            
            type_feature = torch.index_select(
                self.type_embedding, 
                dim=0, 
                index=type_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
        
        else:
            raise ValueError('mode %s not supported' % mode)
        
        score = self.E2Tscore(entity_feature, type_feature)
        
        return score
    
    def E2Tscore(self, entity_feature, type_feature):
        score = torch.matmul(entity_feature, self.M) - type_feature
        score = self.gamma.item() - torch.norm(score, p=2, dim=2)
        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2

        regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def print_accuracy_by_class(model, rank_by_class):
        '''
        Display type names and corresponding accuracies
        '''
        acc_by_class = Counter()
        amount_by_class = {}
        for class_id, rank_list in rank_by_class.items():
            class_name = model.id2type[class_id]
            num_example = len(rank_list)
            acc = np.sum(np.array(rank_list) <= 1)/len(rank_list)
            if num_example > 20: 
                acc_by_class[class_name] = acc
                amount_by_class[class_name] = num_example
        for class_name, acc in acc_by_class.most_common():
            print(class_name+": "+str(acc)+" "+str(amount_by_class[class_name]))

    @staticmethod
    def print_entity_type_pair_and_top_candidates(model):
        '''
        Display the entity name, ground truth type label, top 3 candidates.
        '''
        for rank_list in model.e2t_rank_prediction:
            print('Entity:', model.id2entity[rank_list[0]], 
                  'Ground truth type:', model.id2type[rank_list[1]], 
                  'Top 3 Candidates:', list(map(model.id2type.get, rank_list[2:])))        

    @staticmethod
    def test_step(model, test_pairs, all_true_pairs, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        #Use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
        #Prepare dataloader for evaluation
        test_dataloader = DataLoader(
            E2T_TestDataset(
                test_pairs, 
                all_true_pairs, 
                args.nentity, 
                args.ntype, 
                'choice_batch'
            ), 
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num//2), 
            collate_fn=E2T_TestDataset.collate_fn
        )
        
        logs = []

        step = 0
        total_steps = len(test_dataloader)

        rank_by_class = {}

        with torch.no_grad():
            for positive_sample, negative_sample, filter_bias, mode in test_dataloader:
                if args.cuda:
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    filter_bias = filter_bias.cuda()

                batch_size = positive_sample.size(0)

                score = model((positive_sample, negative_sample), mode)
                
                score += filter_bias

                #Explicitly sort all the entities to ensure that there is no test exposure bias
                argsort = torch.argsort(score, dim = 1, descending=True)

                if mode == 'choice_batch':
                    positive_arg = positive_sample[:, 1]
                else:
                    raise ValueError('mode %s not supported' % mode)

                pos_sample = positive_sample.detach().cpu().numpy()
		
                for i in range(batch_size):
                    model.e2t_rank_prediction.append(pos_sample[i].tolist()+argsort[i, :3].tolist())
                    # ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                    ranking = torch.nonzero(argsort[i, :] == positive_arg[i], as_tuple=False)
                    assert ranking.size(0) == 1

                    #ranking + 1 is the true ranking used in evaluation metrics
                    ranking = 1 + ranking.item()
                    logs.append({
                        'MRR': 1.0/ranking,
                        'MR': float(ranking),
                        'HITS@1': 1.0 if ranking <= 1 else 0.0,
                        'HITS@3': 1.0 if ranking <= 3 else 0.0,
                        'HITS@10': 1.0 if ranking <= 10 else 0.0,
                    })

                    class_id = pos_sample[i, 1]
                    if class_id not in rank_by_class:
                        rank_by_class[class_id] = [ranking]
                    else:
                        rank_by_class[class_id].append(ranking)

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1       

        # model.print_accuracy_by_class(model, rank_by_class)
        # model.print_entity_type_pair_and_top_candidates(model)

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics

