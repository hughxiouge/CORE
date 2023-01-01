#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random
import sys

import pdb

import numpy as np
import torch

from torch.utils.data import DataLoader

from model import KGEModel
from e2t_model import E2TModel
from trt_model import TRTModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator
from e2t_dataloader import E2T_TrainDataset
from e2t_dataloader import E2T_TestDataset
from e2t_dataloader import OneShotIterator
from trt_dataloader import TRT_TrainDataset
from trt_dataloader import TRT_BidirectionalOneShotIterator

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None, 
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')
    
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    parser.add_argument('-dt', '--double_type_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=1000, type=int)
    # parser.add_argument('-t', '--type_dim', default=1000, type=int)
    parser.add_argument('-td', '--type_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-e2tb', '--e2t_batch_size', default=2048, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--e2t_training_steps', default=100000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    
    return parser.parse_args(args)

def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.double_type_embedding = argparse_dict['double_type_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.type_dim = argparse_dict['type_dim']
    args.test_batch_size = argparse_dict['test_batch_size']

def save_all_models(kge_model, e2t_model, trt_model, kge_optimizer, e2t_optimizer, trt_optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': kge_model.state_dict(),
        'e2t_model_state_dict': e2t_model.state_dict(),
        'trt_model_state_dict': trt_model.state_dict(),
        'optimizer_state_dict': kge_optimizer.state_dict(),
        'e2t_optimizer_state_dict': e2t_optimizer.state_dict(),
        'trt_optimizer_state_dict': trt_optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    
    entity_embedding = kge_model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = kge_model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'), 
        relation_embedding
    )

    type_embedding = trt_model.type_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'type_embedding'), 
        type_embedding
    )

    relation_embedding_in_type_space = trt_model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding_in_type_space'), 
        relation_embedding_in_type_space
    )

def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'), 
        relation_embedding
    )

def e2t_save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'e2t_model_state_dic': model.state_dict(),
        'e2t_optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    
    type_embedding = model.type_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'type_embedding'), 
        type_embedding
    )

def trt_save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'trt_model_state_dict': model.state_dict(),
        'trt_optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding_in_type_space'), 
        relation_embedding
    )

def read_pair(file_path, entity2id, type2id):
    '''
    Read triples and map them into ids.
    '''
    pairs = []
    with open(file_path) as fin:
        for line in fin:
            entity, ent_type = line.strip().split('\t')
            pairs.append((entity2id[entity], type2id[ent_type]))
    return pairs

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def get_entity_type(pairs):
    '''
    Build a dictionary of entity's type 
    '''
    entity2type = {}

    for entity, ent_type in pairs:
        if entity not in entity2type:
            entity2type[entity] = []
        entity2type[entity].append(ent_type)
    return entity2type

def get_entity_neighbor(triples, entity2type):
    '''
    Build a dictionary of neighbor entity and relation
    Key: entity; Value: [type id column, rel id column]
    Used to compute inference score from neighboring entity types and relations
    '''

    entity_as_head = {}
    entity_as_tail = {}

    for head, relation, tail in triples:
        if head not in entity_as_head:
            entity_as_head[head] = set([])
        if tail in entity2type:
            tail_types = entity2type[tail]
            for t_type in tail_types:
                entity_as_head[head].add((t_type, relation))
        
        if tail not in entity_as_tail:
            entity_as_tail[tail] = set([])
        if head in entity2type:
            head_types = entity2type[head]
            for h_type in head_types:
                entity_as_tail[tail].add((h_type, relation))
    for head, neighbor_list in entity_as_head.items():
        entity_as_head[head] = np.array(list(entity_as_head[head]))
    for tail, neighbor_list in entity_as_tail.items():
        entity_as_tail[tail] = np.array(list(entity_as_tail[tail]))

    return entity_as_head, entity_as_tail

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

def neighbor_score(model, neighbor, mode, args):
    threshold = 80
    if neighbor.shape[0] > threshold:
        neighbor = neighbor[np.random.choice(neighbor.shape[0], threshold, replace=False)]
    
    neighbor_type = torch.index_select(
            model.type_embedding, 
            dim=0, 
            index=torch.LongTensor(neighbor[:,0].flatten()).cuda()
        ).unsqueeze(0)

    neighbor_relation = torch.index_select(
        model.relation_embedding, 
        dim=0, 
        index=torch.LongTensor(neighbor[:,1].flatten()).cuda()
    ).unsqueeze(0)

    candidate = torch.index_select(
        model.type_embedding, 
        dim=0, 
        index=torch.LongTensor([type_id for type_id in range(model.ntype)]).cuda()
    ).unsqueeze(1)

    model_func = {
        'TransE': model.TransE,
        'DistMult': model.DistMult,
        'ComplEx': model.ComplEx,
        'RotatE': model.RotatE,
        'pRotatE': model.pRotatE
    }

    if mode == 'head-batch':
        score = model_func[args.model](candidate, neighbor_relation, neighbor_type, mode)
    else:
        score = model_func[args.model](neighbor_type, neighbor_relation, candidate, mode)
    return score

def joint_test_step(e2t_model, trt_model, test_pairs, all_true_pairs, args, neighbor_dict):
    '''
    Evaluate the model on test or valid datasets
    '''
    
    e2t_model.eval()
    trt_model.eval()
    
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

    [entity_as_head, entity_as_tail] = neighbor_dict

    rank_by_class = {}

    with torch.no_grad():
        c = 0
        for positive_sample, negative_sample, filter_bias, mode in test_dataloader:
            # print(positive_sample)
            c += 1
            if c % 10 == 0:
                print("Eval batch: ", str(c))
            entity_list = list(positive_sample[:, 0].detach().cpu().numpy())
            if args.cuda:
                positive_sample = positive_sample.cuda()
                negative_sample = negative_sample.cuda()
                filter_bias = filter_bias.cuda()

            batch_size = positive_sample.size(0)

            score = e2t_model((positive_sample, negative_sample), mode)
            score += filter_bias

            as_head_score_list = []
            as_tail_score_list = []
            
            for i in range(batch_size):

                entity = entity_list[i]
                if entity in entity_as_head and len(entity_as_head[entity]) > 0:
                    entity_as_head_neighbor = entity_as_head[entity]
                    entity_as_head_score = neighbor_score(trt_model, entity_as_head_neighbor, 'head-batch', args)
                    entity_as_head_score = torch.mean(entity_as_head_score, dim=1)
                    
                    as_head_score_list.append(entity_as_head_score)

                else:
                    as_head_score_list.append(torch.zeros(trt_model.ntype).cuda())

                if entity in entity_as_tail and len(entity_as_tail[entity]) > 0:
                    entity_as_tail_neighbor = entity_as_tail[entity]
                    entity_as_tail_score = neighbor_score(trt_model, entity_as_tail_neighbor, 'tail-batch', args)
                    entity_as_tail_score = torch.mean(entity_as_tail_score, dim=1)
                    
                    as_tail_score_list.append(entity_as_tail_score)

                else:
                    as_tail_score_list.append(torch.zeros(trt_model.ntype).cuda())

            as_head_score_list = torch.stack(as_head_score_list)
            as_tail_score_list = torch.stack(as_tail_score_list)
            
            
            score += 0.5*(as_head_score_list + filter_bias + as_tail_score_list + filter_bias)    
            
            #Explicitly sort all the entities to ensure that there is no test exposure bias
            argsort = torch.argsort(score, dim = 1, descending=True)

            if mode == 'choice_batch':
                positive_arg = positive_sample[:, 1]
            else:
                raise ValueError('mode %s not supported' % mode)

            pos_sample = positive_sample.detach().cpu().numpy()

            for i in range(batch_size):
                #Notice that argsort is not ranking
                ranking = (argsort[i, :] == positive_arg[i]).nonzero()
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
    
    from collections import Counter
    acc_by_class = Counter()
    amount_by_class = {}
    for class_id, rank_list in rank_by_class.items():
        class_name = e2t_model.id2type[class_id]
        num_example = len(rank_list)
        acc = np.sum(np.array(rank_list) <= 1)/len(rank_list)
        if num_example > 50: 
            acc_by_class[class_name] = acc
            amount_by_class[class_name] = num_example
    for class_name, acc in acc_by_class.most_common():
        print(class_name+": "+str(acc)+" "+str(amount_by_class[class_name]))

    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs])/len(logs)

    return metrics

def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Write logs to checkpoint and console
    set_logger(args)
    
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        id2entity = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            id2entity[int(eid)] = entity

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    
    with open(os.path.join(args.data_path, 'types.dict')) as fin:
        type2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            type2id[relation] = int(rid)

    with open(os.path.join(args.data_path, 'types.dict')) as fin:
        id2type = dict()
        for line in fin:
            tid, type = line.strip().split('\t')
            id2type[int(tid)] = type

    # Read regions for Countries S* datasets
    if args.countries:
        regions = list()
        with open(os.path.join(args.data_path, 'regions.list')) as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions

    nentity = len(entity2id)
    nrelation = len(relation2id)
    ntype = len(type2id)

    args.nentity = nentity
    args.nrelation = nrelation
    args.ntype = ntype

    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    logging.info('#type: %d' % ntype)
    
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))

    train_pairs = read_pair(os.path.join(args.data_path, 'train_type.txt'), entity2id, type2id)
    logging.info('#train: %d' % len(train_pairs))
    valid_pairs = read_pair(os.path.join(args.data_path, 'valid_type.txt'), entity2id, type2id)
    logging.info('#valid: %d' % len(valid_pairs))
    test_pairs = read_pair(os.path.join(args.data_path, 'test_type.txt'), entity2id, type2id)
    logging.info('#test: %d' % len(test_pairs))

    train_trt_triples = read_triple(os.path.join(args.data_path, 'type-relation-type-train.txt'), type2id, relation2id)
    logging.info('#train trt triples: %d' % len(train_trt_triples))
    valid_trt_triples = read_triple(os.path.join(args.data_path, 'type-relation-type-valid.txt'), type2id, relation2id)
    logging.info('#valid trt triples: %d' % len(valid_trt_triples))
    test_trt_triples = read_triple(os.path.join(args.data_path, 'type-relation-type-test.txt'), type2id, relation2id)
    logging.info('#test trt triples: %d' % len(test_trt_triples))
    
    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples
    
    #All type pairs
    all_true_pairs = train_pairs + valid_pairs + test_pairs

    #All trt triples
    all_true_trt_triples = train_trt_triples + valid_trt_triples + test_trt_triples

    #Get dictionary of entity's types
    entity2type = get_entity_type(train_pairs)

    #Inference from neighbor type and relation
    entity_as_head, entity_as_tail = get_entity_neighbor(train_triples, entity2type)

    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )

    e2t_model = E2TModel(
        nentity=nentity, 
        ntype=ntype, 
        entity_dim=args.hidden_dim, 
        type_dim=args.type_dim, 
        gamma=args.gamma, 
        entity_parameter=kge_model.entity_embedding,
        id2type=id2type,
        id2entity=id2entity,
        double_entity_embedding=args.double_entity_embedding,
        double_type_embedding=args.double_type_embedding
    )

    trt_model = TRTModel(
        model_name=args.model,
        ntype=ntype, 
        nrelation=nrelation, 
        # hidden_dim=int(args.type_dim/2),
        hidden_dim=args.type_dim,
        gamma=args.gamma, 
        type_parameter=e2t_model.type_embedding,
        double_type_embedding=args.double_type_embedding,
        double_relation_embedding=args.double_relation_embedding
    )
    
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    for name, param in e2t_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    for name, param in trt_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()
        e2t_model = e2t_model.cuda()
        trt_model = trt_model.cuda()
    
    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        e2t_train_dataloader = DataLoader(
            E2T_TrainDataset(train_pairs, nentity, ntype, args.negative_sample_size, 'choice_batch'),
            batch_size=args.e2t_batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=E2T_TrainDataset.collate_fn
        )

        e2t_train_iterator = OneShotIterator(e2t_train_dataloader)

        trt_train_dataloader_head = DataLoader(
            TRT_TrainDataset(train_trt_triples, ntype, nrelation, args.negative_sample_size, 'head-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TRT_TrainDataset.collate_fn
        )
        
        trt_train_dataloader_tail = DataLoader(
            TRT_TrainDataset(train_trt_triples, ntype, nrelation, args.negative_sample_size, 'tail-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TRT_TrainDataset.collate_fn
        )
        
        trt_train_iterator = BidirectionalOneShotIterator(trt_train_dataloader_head, trt_train_dataloader_tail)

        # Set training configuration
        current_learning_rate = args.learning_rate
        kge_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()), 
            lr=current_learning_rate
        )

        e2t_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, e2t_model.parameters()), 
            lr=current_learning_rate
        )

        trt_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, trt_model.parameters()), 
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        e2t_model.load_state_dict(checkpoint['e2t_model_state_dict'])
        trt_model.load_state_dict(checkpoint['trt_model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            kge_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            e2t_optimizer.load_state_dict(checkpoint['e2t_optimizer_state_dict'])
            trt_optimizer.load_state_dict(checkpoint['trt_optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0
    
    step = init_step
    
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('e2t_batch_size = %d' % args.e2t_batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)

        training_logs = []
        e2t_training_logs = []
        trt_training_logs = []

        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }

        save_all_models(kge_model, e2t_model, trt_model, kge_optimizer, e2t_optimizer, trt_optimizer, save_variable_list, args)

        #Training Loop
        steps_each_epoch = 1000
        num_epoch = int(args.max_steps/steps_each_epoch)
        
        for epoch in range(num_epoch):
            logging.info('Starting epoch ' + str(epoch))
            logging.info('Training '+args.model+' epoch'+str(epoch))
            if epoch > int(args.e2t_training_steps/steps_each_epoch):
                logging.info('Training TRT epoch'+str(epoch))
                for step in range(init_step+epoch*steps_each_epoch, init_step+(epoch+1)*steps_each_epoch):
                    log = trt_model.train_step(trt_model, trt_optimizer, trt_train_iterator, args)
                    
                    trt_training_logs.append(log)
                        
                    if step % args.log_steps == 0:
                        metrics = {}
                        for metric in trt_training_logs[0].keys():
                            metrics[metric] = sum([log[metric] for log in trt_training_logs])/len(trt_training_logs)
                        log_metrics('Training average', step, metrics)
                        trt_training_logs = []
                        
                    if args.do_valid and (step+1) % (args.valid_steps) == 0:
                        logging.info('Evaluating on TRT Valid Dataset...')
                        metrics = trt_model.test_step(trt_model, valid_trt_triples, all_true_trt_triples, args)
                        log_metrics('Valid', step, metrics)

                        logging.info('Evaluating on TRT Test Dataset...')
                        metrics = trt_model.test_step(trt_model, test_trt_triples, all_true_trt_triples, args)
                        log_metrics('Test', step, metrics)
            
            else:
                logging.info('Training KGE epoch'+str(epoch))
                for step in range(init_step+epoch*steps_each_epoch, init_step+(epoch+1)*steps_each_epoch):
                    log = kge_model.train_step(kge_model,kge_optimizer, train_iterator, args)
                    
                    training_logs.append(log)
                    
                    if step >= warm_up_steps:
                        current_learning_rate = current_learning_rate / 10
                        logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                        optimizer = torch.optim.Adam(
                            filter(lambda p: p.requires_grad, kge_model.parameters()), 
                            lr=current_learning_rate
                        )
                        warm_up_steps = warm_up_steps * 3
                        
                    if step % args.log_steps == 0:
                        metrics = {}
                        for metric in training_logs[0].keys():
                            metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                        log_metrics('Training average', step, metrics)
                        training_logs = []
                        
                    if args.do_valid and (step+1) % args.valid_steps == 0:
                        logging.info('Evaluating on KGE Valid Dataset...')
                        metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
                        log_metrics('Valid', step, metrics)
                
                # for E2T
                logging.info('Training E2T epoch'+str(epoch))
                e2t_model.entity_embedding.data = kge_model.entity_embedding.detach().clone()
                e2t_model.type_embedding.data = trt_model.type_embedding.detach().clone()
                for step in range(init_step+epoch*steps_each_epoch, init_step+(epoch+1)*steps_each_epoch):
                    log = e2t_model.train_step(e2t_model, e2t_optimizer, e2t_train_iterator, args)
                    
                    e2t_training_logs.append(log)
                    
                    if step >= warm_up_steps:
                        current_learning_rate = current_learning_rate / 10
                        logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                        e2t_optimizer = torch.optim.Adam(
                            filter(lambda p: p.requires_grad, e2t_model.parameters()), 
                            lr=current_learning_rate
                        )
                        warm_up_steps = warm_up_steps * 3
                        
                    if step % args.log_steps == 0:
                        metrics = {}
                        for metric in e2t_training_logs[0].keys():
                            metrics[metric] = sum([log[metric] for log in e2t_training_logs])/len(e2t_training_logs)
                        log_metrics('Training average', step, metrics)
                        e2t_training_logs = []
                        
                    if args.do_valid and (step+1) % (args.valid_steps) == 0:
                        logging.info('Evaluating on Type Prediction Valid Dataset...')
                        metrics = e2t_model.test_step(e2t_model, valid_pairs, all_true_pairs, args)
                        log_metrics('Valid', step, metrics)
                
                # for TRT
                logging.info('Training TRT epoch'+str(epoch))
                trt_model.type_embedding.data = e2t_model.type_embedding.detach().clone()
                for step in range(init_step+epoch*steps_each_epoch, init_step+(epoch+1)*steps_each_epoch):
                    log = trt_model.train_step(trt_model, trt_optimizer, trt_train_iterator, args)
                    
                    trt_training_logs.append(log)
                    
                    if step >= warm_up_steps:
                        current_learning_rate = current_learning_rate / 10
                        logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                        trt_optimizer = torch.optim.Adam(
                            filter(lambda p: p.requires_grad, trt_model.parameters()), 
                            lr=current_learning_rate
                        )
                        warm_up_steps = warm_up_steps * 3
                        
                    if step % args.log_steps == 0:
                        metrics = {}
                        for metric in trt_training_logs[0].keys():
                            metrics[metric] = sum([log[metric] for log in trt_training_logs])/len(trt_training_logs)
                        log_metrics('Training average', step, metrics)
                        trt_training_logs = []
                        
                    if args.do_valid and (step+1) % (args.valid_steps) == 0:
                        logging.info('Evaluating on TRT Valid Dataset...')
                        metrics = trt_model.test_step(trt_model, valid_trt_triples, all_true_trt_triples, args)
                        log_metrics('Valid', step, metrics)

            save_variable_list = {
                'step': step, 
                'current_learning_rate': current_learning_rate,
                'warm_up_steps': warm_up_steps
            }

            save_all_models(kge_model, e2t_model, trt_model, kge_optimizer, e2t_optimizer, trt_optimizer, save_variable_list, args)


    if args.do_valid:
        logging.info('Evaluating on Type Prediction Valid Dataset...')
        metrics = e2t_model.test_step(e2t_model, valid_pairs, all_true_pairs, args)
        log_metrics('Valid', step, metrics)
    
    if args.do_test:
        logging.info('Evaluating on Type Prediction Test Dataset...')
        metrics = e2t_model.test_step(e2t_model, test_pairs, all_true_pairs, args)
        log_metrics('Test', step, metrics)

    if args.evaluate_train:
        logging.info('Evaluating on KGE Training Dataset...')
        metrics = kge_model.test_step(kge_model, train_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)

        logging.info('Evaluating on Type Prediction Training Dataset...')
        metrics = e2t_model.test_step(e2t_model, train_pairs, all_true_pairs, args)
        log_metrics('Test', step, metrics)

        logging.info('Evaluating on TRT Training Dataset...')
        metrics = trt_model.test_step(trt_model, train_trt_triples, all_true_trt_triples, args)
        log_metrics('Test', step, metrics)
        
if __name__ == '__main__':
    main(parse_args())
