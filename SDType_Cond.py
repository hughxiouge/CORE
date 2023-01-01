from collections import Counter

insnet_train_file = 'data/DBPEDIA-clean/train.txt'
insnet_test_file = 'data/DBPEDIA-clean/test.txt'
type_train_file = 'data/DBPEDIA-clean/train_type.txt'
type_test_file = 'data/DBPEDIA-clean/test_type.txt'


def build_adjacency_dict(file_name):
    adjacency_dict = {}
    with open(file_name, 'r') as f:
        for line in f.read().splitlines():
            head, relation, tail = line.split()
            if head not in adjacency_dict:
                adjacency_dict[head] = {relation: [tail]}
            else:
                if relation not in adjacency_dict[head]:
                    adjacency_dict[head][relation] = [tail]
                else:
                    adjacency_dict[head][relation].append(tail)
            if tail not in adjacency_dict:
                adjacency_dict[tail] = {'inv_'+relation: [head]}
            else:
                if 'inv_'+relation not in adjacency_dict[tail]:
                    adjacency_dict[tail]['inv_'+relation] = [head]
                else:
                    adjacency_dict[tail]['inv_'+relation].append(head)
    return adjacency_dict

train_adjacency_dict = build_adjacency_dict(insnet_train_file)
test_adjacency_dict = build_adjacency_dict(insnet_test_file)

def build_label_dict(file_name):
    type_label_dict = {}
    with open(file_name, 'r') as f:
        for line in f.read().splitlines():
            entity, type_label = line.split()
            if entity not in type_label_dict:
                type_label_dict[entity] = set([type_label])
            else:
                type_label_dict[entity].add(type_label) 
    return type_label_dict

train_type_label_dict = build_label_dict(type_train_file)
test_type_label_dict = build_label_dict(type_test_file)

def get_relation_head_tail_entity_dict(file_name):
    relation_head_tail_entity = {}
    with open(file_name, 'r') as f:
        for line in f.read().splitlines():
            head, relation, tail = line.split()
            if relation not in relation_head_tail_entity:
                relation_head_tail_entity[relation] = [set([head]), set([tail])]
            else:
                relation_head_tail_entity[relation][0].add(head)
                relation_head_tail_entity[relation][1].add(tail)
    return relation_head_tail_entity

def get_relation_head_tail_type_dict(relation_head_tail_entity):
    relation_head_tail_type = {}
    for relation, head_tail_entity in relation_head_tail_entity.items():
        head_set, tail_set = head_tail_entity
        head_type, tail_type = set(), set()
        for h in head_set:
            if h in train_type_label_dict:
                head_type = head_type.union(train_type_label_dict[h])
        for t in tail_set:
            if t in train_type_label_dict:
                tail_type = tail_type.union(train_type_label_dict[t])
        relation_head_tail_type[relation] = [head_type, tail_type]
    return relation_head_tail_type

relation_head_tail_entity = get_relation_head_tail_entity_dict(insnet_train_file)
relation_head_tail_type = get_relation_head_tail_type_dict(relation_head_tail_entity)

def get_relation_head_tail_type_count_dict(all_relations, file_name, train_type_label_dict):
    relation_head_tail_type_count = {}
    for relation in all_relations:
        relation_head_tail_type_count[relation] = [{},{}]
    with open(file_name, 'r') as f:
        for line in f.read().splitlines():
            head, relation, tail = line.split()
            if head in train_type_label_dict:
                head_type_set = train_type_label_dict[head]
                for head_type in head_type_set:
                    if head_type not in relation_head_tail_type_count[relation][0]:
                        relation_head_tail_type_count[relation][0][head_type] = 1
                    else:
                        relation_head_tail_type_count[relation][0][head_type] += 1
            if tail in train_type_label_dict:    
                tail_type_set = train_type_label_dict[tail]
                for tail_type in tail_type_set:
                    if tail_type not in relation_head_tail_type_count[relation][1]:
                        relation_head_tail_type_count[relation][1][tail_type] = 1
                    else:
                        relation_head_tail_type_count[relation][1][tail_type] += 1
    return relation_head_tail_type_count

all_relations = relation_head_tail_entity.keys()
relation_head_tail_type_count = get_relation_head_tail_type_count_dict(all_relations, insnet_train_file, train_type_label_dict)

def get_conditional_count(all_relations, relation_head_tail_type, file_name, train_type_label_dict):
    tail_type_count_given_head_type = {}
    head_type_count_given_tail_type = {}
    for relation in all_relations:
        head_type_set, tail_type_set = relation_head_tail_type[relation]
        tail_type_count_given_head_type[relation] = {'Unknown':{'Unknown':0}} 
        head_type_count_given_tail_type[relation] = {'Unknown':{'Unknown':0}}
        # add smoothing? 
        for tail_type in tail_type_set:
            tail_type_count_given_head_type[relation]['Unknown'][tail_type] = 1
        for head_type in head_type_set:
            head_type_count_given_tail_type[relation]['Unknown'][head_type] = 1
        for head_type in head_type_set:
            tail_type_count_given_head_type[relation][head_type] = {'Unknown':0}
            for tail_type in tail_type_set:
                tail_type_count_given_head_type[relation][head_type][tail_type] = 1
        for tail_type in tail_type_set:
            head_type_count_given_tail_type[relation][tail_type] = {'Unknown':0}
            for head_type in head_type_set:
                head_type_count_given_tail_type[relation][tail_type][head_type] = 1
    with open(file_name, 'r') as f:
        for line in f.read().splitlines():
            head, relation, tail = line.split()
            if head in train_type_label_dict:
                head_type_set = train_type_label_dict[head]
            else:
                head_type_set = set(['Unknown'])
            if tail in train_type_label_dict:
                tail_type_set = train_type_label_dict[tail]
            else:
                tail_type_set = set(['Unknown'])
            for head_type in head_type_set:
                for tail_type in tail_type_set:
                    tail_type_count_given_head_type[relation][head_type][tail_type] += 1
                    head_type_count_given_tail_type[relation][tail_type][head_type] += 1
    return tail_type_count_given_head_type, head_type_count_given_tail_type

tail_type_count_given_head_type, head_type_count_given_tail_type = get_conditional_count(all_relations, 
                                                                                          relation_head_tail_type, 
                                                                                          insnet_train_file, 
                                                                                          train_type_label_dict)
def normalize(histogram):
    prob_value = [x/sum(histogram.values()) for x in histogram.values()]
    probability = dict(zip(histogram.keys(), prob_value))
    return probability

# def normalize(histogram):
#     prob_value = [x/sum(histogram.values()) for x in histogram.values()]
#     return prob_value

type_case_counter = 0
hit1 = 0
hit3 = 0
hit10 = 0
MRR = 0
for ent, type_label_set in test_type_label_dict.items():
    count = Counter({})
    if ent in train_adjacency_dict:
        for rel, given_neighbors_list in train_adjacency_dict[ent].items():
            # Forward relation, head is to be infered, given tail
            if 'inv_' not in rel:
                for tail in given_neighbors_list:
                    if tail in train_type_label_dict:
                        tail_type_set = train_type_label_dict[tail]
                    else:
                        tail_type_set = set(['Unknown'])
                    for tail_type in tail_type_set:
                        histogram = head_type_count_given_tail_type[rel][tail_type]
                        probability = normalize(histogram)
                        count += probability
            # Backward relation, tail is to be inferred, given head
            else:
                rel = rel.split('_')[1]
                for head in given_neighbors_list:
                    if head in train_type_label_dict:
                        head_type_set = train_type_label_dict[head]
                    else:
                        head_type_set = set(['Unknown'])
                    for head_type in head_type_set:
                        histogram = tail_type_count_given_head_type[rel][head_type]
                        probability = normalize(histogram)
                        count += probability
        del count['Unknown']
        type_label_candidate_rank = [type_count[0] for type_count in count.most_common()]        
        for type_label in type_label_set:
            type_case_counter += 1
            try:
                rank = type_label_candidate_rank.index(type_label)+1    
                MRR += 1/rank
                if rank == 1:
                    hit1 += 1
                if rank <= 3:
                    hit3 += 1
                if rank <= 10:
                    hit10 += 1
            except:
                MRR += 0.0001
print("MRR:", MRR/type_case_counter)
print("Hit@1:", 100*hit1/type_case_counter)
print("Hit@3:", 100*hit3/type_case_counter)
print("Hit@10:", 100*hit10/type_case_counter)