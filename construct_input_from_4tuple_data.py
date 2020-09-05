import json
import itertools
from transformers import BertTokenizer
import torch 
import re
import collections
from operator import itemgetter


def compare_time_helper(format_time_1,format_time_2):
    year_1,month_1,day_1 = format_time_1.split('-')
    year_2,month_2,day_2 = format_time_2.split('-')
    if int(year_1) == int(year_2):
        if int(month_1) == int(month_2):
            if int(day_1) == int(day_2):
                return 0
            elif int(day_1) < int(day_2):
                return -1
            else:
                return 1
        elif int(month_1) < int(month_2):
            return -1
        else:
            return 1
    elif int(year_1) < int(year_2):
        return -1
    else:
        return 1

def compare_time(four_tuple_1,four_tuple_2):
    '''only use start info'''
    if four_tuple_1[0] != '-inf' and four_tuple_2[0] != '-inf':
        if compare_time_helper(four_tuple_1[0],four_tuple_2[0]) == 0:
            if four_tuple_1[1] != '+inf' and four_tuple_2[1] != '+inf':
                return compare_time_helper(four_tuple_1[1],four_tuple_2[1])
            else:
                return 0
        elif compare_time_helper(four_tuple_1[0],four_tuple_2[0]) == -1:
            return -1
        else:
            return 1

    else:
        if four_tuple_1[0] != '-inf':
            format_time_1 = four_tuple_1[0]
        else:
            format_time_1 = four_tuple_1[1]
        
        if four_tuple_2[0] != '-inf':
            format_time_2 = four_tuple_2[0]
        else:
            format_time_2 = four_tuple_2[1]
        return compare_time_helper(format_time_1,format_time_2)

def construct_cluster(sorted_ems,em_to_four_tuple_dict):
    cluster_list = []
    local_cluster = []
    for em in sorted_ems:
        if local_cluster == []:
            local_cluster.append(em)
        else:
            if em_to_four_tuple_dict[em][0] == em_to_four_tuple_dict[local_cluster[-1]][0] and   em_to_four_tuple_dict[em][1] == em_to_four_tuple_dict[local_cluster[-1]][1]:
                local_cluster.append(em)
            else:
                cluster_list.append(local_cluster)
                local_cluster = []
                local_cluster.append(em)
    if local_cluster != []:
        cluster_list.append(local_cluster)
    return cluster_list


def construct_input(cluster_list):
    ## dead tree approach
        # nodes = []
        # for i in range(len(cluster_list)):
        #     this_cluster = cluster_list[i]
        #     subroot = [TreeNode(this_em) for this_em in this_cluster]
        #     nodes.append(subroot)
        # roots = nodes[0]
        # for i in range(len(nodes)-1):
        #     this_layer = nodes[i]
        #     next_layer = nodes[i+1]
        #     for node in this_layer:
        #         node.children = next_layer
        

        
        # # print([r.value for r in roots])

        # def bfs_return_paths(nodes):
        #     all_paths = []
        #     for roots in nodes:
        #         for root in roots:
        #             is_visited = {}
        #             path_dict = {} # key: node , value: all_paths that can reach this node
        #             q = []
        #             q.append(root)
        #             path_dict[root.value] = set()
        #             is_visited[root.value] = True
        #             paths = []
                    
        #             while q != []:
        #                 this_node = q[0]
        #                 q.pop(0)
        #                 new_paths = [item for item in paths]
        #                 # print('before: ',new_paths)
        #                 # print('')
                        
        #                 for nb in this_node.children:
        #                     if nb not in is_visited:
        #                         q.append(nb)
        #                         is_visited[nb.value] = True
        #                     new_paths.append([this_node.value,nb.value])
        #                     if nb not in path_dict:
        #                         path_dict[nb.value] = set()
        #                     path_dict[nb.value].add(tuple([this_node.value,nb.value]))
        #                     # print(new_paths)
                            
        #                     for p in path_dict[this_node.value]:
        #                         pp = list(p).copy()
        #                         pp.append(nb.value)
        #                         new_paths.append(pp)
        #                         path_dict[nb.value].add(tuple(pp))
        #                         # print('\t',pp)
        #                 paths = new_paths.copy()
        #                 # print('after: ',paths)
        #                 # print('')
        #                 # print('===================')
        #             for p in paths:
        #                 all_paths.append(p)
        #     paths_set = set()
        #     for l in all_paths:
        #         paths_set.add(tuple(l))
        #     return paths_set
        
        # return bfs_return_paths(nodes[:-1])
    all_paths = []
    for size in range(2,len(cluster_list)+1):
        for i in range(0,len(cluster_list)+1-size):
            a = [cluster_list[j] for j in range(i,i+size)]
            pairs = list(itertools.product(*a))
            all_paths += pairs
    return all_paths

def construct_input_pair_helper(ems,em_to_four_tuple_dict):
    #ems: list of event mention ids
    
    def bubbleSort(arr): 
        n = len(arr) 
        for i in range(n): 
            swapped = False
            for j in range(0, n-i-1): 
                if compare_time(em_to_four_tuple_dict[arr[j]],em_to_four_tuple_dict[arr[j+1]]) == 1: 
                    arr[j], arr[j+1] = arr[j+1], arr[j] 
                    swapped = True
            if swapped == False: 
                break

    to_be_removed = []
    for em in ems:
        if em not in em_to_four_tuple_dict:
            to_be_removed.append(em)
        else:
            em_time = em_to_four_tuple_dict[em]
            if em_time[0] == '-inf' and em_time[1] == '+inf':
                to_be_removed.append(em)
    for em in to_be_removed:
        ems.remove(em)
    
    """sorting"""
    bubbleSort(ems)
    # print('')
    # print('after sorting')
    # print('')
    # for em in ems:
    #     print(em,em_to_four_tuple_dict[em])

    """clustering"""
    cluster_list = construct_cluster(ems,em_to_four_tuple_dict)
    # print('')
    # print('after clustering')
    # print('')
    # for cluster in cluster_list:
    #     print(cluster)

    if len(cluster_list) == 1:
        return []
    else:
        """constructing input"""
        # print('')
        # print('after constructed input')
        # print('')
        return_set = construct_input(cluster_list)
        # print('length of return_set: ', len(return_set),'\n')
        # print(return_set)

        return return_set


def construct_input_pair(tokenizer,input_pairs = None):

    '''set up arg_entity_to_ems_dict'''
    data = {}
    with open('em_id_lookup.json') as f:
        data = json.load(f)

    arg_em_dict = {}
    for key,value in data.items():
        if value['arguments'] != []:
            for entity in value['arguments']:
                sep = '-'
                en_id = sep.join(entity['entity_id'].split('-')[:-1])
                if en_id not in arg_em_dict:
                    arg_em_dict[en_id] = []
                    arg_em_dict[en_id].append({'event_mention':key,'role':entity['role']})
                else:
                    arg_em_dict[en_id].append({'event_mention':key,'role':entity['role']})
    count = 0
    for entity_key, ems in arg_em_dict.items():
        if len(ems) >= 2:
            count+=1
    print('num of arg entity with appeared in more than one event mention:',count)


    '''set up em to 4 tuple time dict'''
    em_to_four_tuple_dict = {}
    with open('4_tuple_upload.json') as f:
        data = json.load(f)
    for item in data:
        key = item['event_mention']
        value = item['four_tuple']
        if key in em_to_four_tuple_dict:
            if value != em_to_four_tuple_dict[key]:
                print(value)
                print(em_to_four_tuple_dict[key])
        else:
            em_to_four_tuple_dict[key] = value

    if input_pairs is None:
        print_count = 0
        input_pairs = []
        for entity_key, ems in arg_em_dict.items():
            if len(ems) >= 2:
                if print_count<=3:
                    print('==============================================\n')
                    for em in ems:
                        print(em,em_to_four_tuple_dict[em['event_mention']])
                    input_ems = [em['event_mention'] for em in ems]
                    pairs = construct_input_pair_helper(input_ems,em_to_four_tuple_dict)
                    print('')
                    print('length of pairs',len(pairs))
                    print('')
                    print(pairs)
                    print('==============================================\n')
                    print_count += 1
                else:
                    input_ems = [em['event_mention'] for em in ems]
                    pairs = construct_input_pair_helper(input_ems,em_to_four_tuple_dict)

                for p in pairs:
                    input_pairs.append(p)

    with open('ACE05_events_three_level_train_emid_lookup.json') as f:
        lookup_dict_whole =  json.load(f)
    with open('ACE05_events_three_level_dev_emid_lookup.json') as f:
        lookup_dict_dev =  json.load(f)
    with open('ACE05_events_three_level_test_emid_lookup.json') as f:
        lookup_dict_test =  json.load(f)
    
    # print(len(lookup_dict_whole))
    # print(len(lookup_dict_dev))
    # print(len(lookup_dict_test))

    lookup_dict_whole.update(lookup_dict_dev)
    lookup_dict_whole.update(lookup_dict_test)

    # print(len(lookup_dict_whole))

    input_items = []
    for pair in input_pairs:
        # print(pair)
        historical_event_types = [lookup_dict_whole[pair[i]]['EVENT_SUBTYPE'] for i in range(len(pair)-1)]

        label = lookup_dict_whole[pair[-1]]['EVENT_SUBTYPE']
        instance_sents = [lookup_dict_whole[pair[i]]['INSTANCE_LEVEL'] for i in range(len(pair)-1)]
        role_sents = [lookup_dict_whole[pair[i]]['ROLE_TYPE_LEVEL'] for i in range(len(pair)-1)]
        entity_sents = [lookup_dict_whole[pair[i]]['ENTITY_TYPE_LEVEL'] for i in range(len(pair)-1)]
        
        sep_token = tokenizer.sep_token
        first_sent_instance = sep_token.join(instance_sents)
        first_sent_role = sep_token.join(role_sents)
        first_sent_entity = sep_token.join(entity_sents)
        item = {}
        item['first_sentence_instance'] = first_sent_instance
        item['first_sentence_role'] = first_sent_role
        item['first_sentence_entity'] = first_sent_entity
        item['label'] = label
        item['historical_event_types'] = historical_event_types
        input_items.append(item)
        # print(item)
        # print('')
        # print('=====================================')
    print('total num of input items:',len(input_items))
    # for i in range(10):
    #     print('')
    #     print(input_items[i])
    #     print('')
    #     print('=====================')
    #     print('=====================')
    return input_items

# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# construct_input_pair(tokenizer)

## output
# with open('arg_entity_to_ems.json','w') as out_file:
#     json.dump(arg_em_dict, out_file, indent = 4, sort_keys = False) 
def construct_input_pair_individual(em_pair, tokenizer):
    '''set up arg_entity_to_ems_dict'''
    data = {}
    with open('em_id_lookup.json') as f:
        data = json.load(f)

    arg_em_dict = {}
    for key,value in data.items():
        if value['arguments'] != []:
            for entity in value['arguments']:
                sep = '-'
                en_id = sep.join(entity['entity_id'].split('-')[:-1])
                if en_id not in arg_em_dict:
                    arg_em_dict[en_id] = []
                    arg_em_dict[en_id].append({'event_mention':key,'role':entity['role']})
                else:
                    arg_em_dict[en_id].append({'event_mention':key,'role':entity['role']})
    count = 0
    for entity_key, ems in arg_em_dict.items():
        if len(ems) >= 2:
            count+=1
    print('num of arg entity with appeared in more than one event mention:',count)


    '''set up em to 4 tuple time dict'''
    em_to_four_tuple_dict = {}
    with open('4_tuple_upload.json') as f:
        data = json.load(f)
    for item in data:
        key = item['event_mention']
        value = item['four_tuple']
        if key in em_to_four_tuple_dict:
            if value != em_to_four_tuple_dict[key]:
                print(value)
                print(em_to_four_tuple_dict[key])
        else:
            em_to_four_tuple_dict[key] = value
    
    with open('ACE05_events_three_level_train_emid_lookup.json') as f:
        lookup_dict_whole =  json.load(f)
    with open('ACE05_events_three_level_dev_emid_lookup.json') as f:
        lookup_dict_dev =  json.load(f)
    with open('ACE05_events_three_level_test_emid_lookup.json') as f:
        lookup_dict_test =  json.load(f)
    
    # print(len(lookup_dict_whole))
    # print(len(lookup_dict_dev))
    # print(len(lookup_dict_test))

    lookup_dict_whole.update(lookup_dict_dev)
    lookup_dict_whole.update(lookup_dict_test)

    # print(len(lookup_dict_whole))

    pair = em_pair
    print(pair)
    historical_event_types = [lookup_dict_whole[pair[i]]['EVENT_SUBTYPE'] for i in range(len(pair)-1)]
    label = lookup_dict_whole[pair[-1]]['EVENT_SUBTYPE']
    instance_sents = [lookup_dict_whole[pair[i]]['INSTANCE_LEVEL'] for i in range(len(pair)-1)]
    role_sents = [lookup_dict_whole[pair[i]]['ROLE_TYPE_LEVEL'] for i in range(len(pair)-1)]
    entity_sents = [lookup_dict_whole[pair[i]]['ENTITY_TYPE_LEVEL'] for i in range(len(pair)-1)]
    
    sep_token = tokenizer.sep_token
    first_sent_instance = sep_token.join(instance_sents)
    first_sent_role = sep_token.join(role_sents)
    first_sent_entity = sep_token.join(entity_sents)
    item = {}
    item['first_sentence_instance'] = first_sent_instance
    item['first_sentence_role'] = first_sent_role
    item['first_sentence_entity'] = first_sent_entity
    item['historical_event_types'] = historical_event_types
    item['label'] = label
    # print(item)
    # print('')
    # print('=====================================')
    # print('total num of input items:',len(input_items))
    # for i in range(10):
    #     print('')
    #     print(input_items[i])
    #     print('')
    #     print('=====================')
    #     print('=====================')
    return item