from transformers import BertTokenizer, BertForMaskedLM
import torch 
import json
import re
import collections
from operator import itemgetter



"""set up type dicts"""
# raw_file = 'all_files_update.json'
# event_type_dict,entity_type_dict,role_type_dict = gen_event_entity_role_dict(raw_file)
# class_size = len(event_type_dict)

event_type_dict = {'Divorce': 0, 'EndPosition': 1, 'Acquit': 2, 'Meet': 3, 'Die': 4, 'Extradite': 5, 'Sue': 6, 'Elect': 7, 'Convict': 8, 'TransferOwnership': 9, 'Marry': 10, 'Attack': 11, 'StartPosition': 12, 'ArrestJail': 13, 'ReleaseParole': 14, 'Nominate': 15, 'Transport': 16, 'Fine': 17, 'Sentence': 18, 'TrialHearing': 19, 'BeBorn': 20, 'Pardon': 21, 'Demonstrate': 22, 'Execute': 23, 'StartOrg': 24, 'PhoneWrite': 25, 'Appeal': 26, 'Injure': 27, 'ChargeIndict': 28, 'TransferMoney': 29, 'EndOrg':30, 'DeclareBankruptcy':31, 'MergeOrg':32,'Null':33}

entity_type_dict = {'ORG': 0, 'LOC': 1, 'VEH': 2, 'WEA': 3, 'GPE': 4, 'FAC': 5, 'PER': 6, 'CLS':7, 'SEP':8, 'PAD':9, 'UNK':10}

for e in entity_type_dict.keys():
    entity_type_dict[e] += 1

role_type_dict = {'Target': 0, 'Plaintiff': 1, 'Person': 2, 'Seller': 3, 'Time': 4, 'Recipient': 5, 'Instrument': 6, 'Artifact': 7, 'Adjudicator': 8, 'Prosecutor': 9, 'Agent': 10, 'Beneficiary': 11, 'Attacker': 12, 'Victim': 13, 'Money': 14, 'Buyer': 15, 'Docid': 16, 'Crime': 17, 'Giver': 18, 'Sentence': 19, 'Org': 20, 'Defendant': 21, 'Position': 22, 'Vehicle': 23, 'Destination': 24, 'Origin': 25, 'Place': 26, 'Entity': 27, 'Price':28, 'CLS':29, 'SEP':30, 'PAD':31} 

for r in role_type_dict.keys():
    role_type_dict[r] += 1

entity_type_dict['OTHER'] = 0
role_type_dict['Other'] = 0

entity_type_dict_new = {}
role_type_dict_new = {}

for et in entity_type_dict.keys():
    if et == 'CLS' or et =='SEP' or et=='PAD' or et=='OTHER':
        pass
    else:
        et_b = 'B-'+et
        et_i = 'I-'+et
        entity_type_dict_new[et_b] = len(entity_type_dict_new)
        entity_type_dict_new[et_i] = len(entity_type_dict_new)

for rt in role_type_dict.keys():
    if rt == 'CLS' or rt =='SEP' or rt =='PAD' or rt=='Other':
        pass
    else:
        rt_b = 'B-'+rt
        rt_i = 'I-'+rt
        role_type_dict_new[rt_b] = len(role_type_dict_new)
        role_type_dict_new[rt_i] = len(role_type_dict_new)

entity_type_dict = {'B-ORG': 0, 'I-ORG': 1, 'B-LOC': 2, 'I-LOC': 3, 'B-VEH': 4, 'I-VEH': 5, 'B-WEA': 6, 'I-WEA': 7, 'B-GPE': 8, 'I-GPE': 9, 'B-FAC': 10, 'I-FAC': 11, 'B-PER': 12, 'I-PER': 13, 'B-UNK': 14, 'I-UNK': 15,'CLS':16,'SEP':17,'PAD':18,'OTHER':19}
role_type_dict = {'B-Target': 0, 'I-Target': 1, 'B-Plaintiff': 2, 'I-Plaintiff': 3, 'B-Person': 4, 'I-Person': 5, 'B-Seller': 6, 'I-Seller': 7, 'B-Time': 8, 'I-Time': 9, 'B-Recipient': 10, 'I-Recipient': 11, 'B-Instrument': 12, 'I-Instrument': 13, 'B-Artifact': 14, 'I-Artifact': 15, 'B-Adjudicator': 16, 'I-Adjudicator': 17, 'B-Prosecutor': 18, 'I-Prosecutor': 19, 'B-Agent': 20, 'I-Agent': 21, 'B-Beneficiary': 22, 'I-Beneficiary': 23, 'B-Attacker': 24, 'I-Attacker': 25, 'B-Victim': 26, 'I-Victim': 27, 'B-Money': 28, 'I-Money': 29, 'B-Buyer': 30, 'I-Buyer': 31, 'B-Docid': 32, 'I-Docid': 33, 'B-Crime': 34, 'I-Crime': 35, 'B-Giver': 36, 'I-Giver': 37, 'B-Sentence': 38, 'I-Sentence': 39, 'B-Org': 40, 'I-Org': 41, 'B-Defendant': 42, 'I-Defendant': 43, 'B-Position': 44, 'I-Position': 45, 'B-Vehicle': 46, 'I-Vehicle': 47, 'B-Destination': 48, 'I-Destination': 49, 'B-Origin': 50, 'I-Origin': 51, 'B-Place': 52, 'I-Place': 53, 'B-Entity': 54, 'I-Entity': 55, 'B-Price': 56, 'I-Price': 57,'CLS':58,'SEP':59,'PAD':60,'Other':61}

idx_to_event = {value:key for key,value in event_type_dict.items()}
idx_to_role = {value:key for key,value in role_type_dict.items()}
idx_to_entity = {value:key for key,value in entity_type_dict.items()}






RESTRICTION_SPAN = 3
# doc_id_to_time_json = 'doc_id_to_time.json'
def sort_by_time(input_data, doc_id_to_time_json):
    with open(doc_id_to_time_json) as f:
        doc_id_to_time = json.load(f)
    new_list = []
    for item in input_data:
        doc_id = item['DOC_ID']
        year = doc_id_to_time[doc_id]['year']
        month = doc_id_to_time[doc_id]['month']
        day = doc_id_to_time[doc_id]['day']
        h = doc_id_to_time[doc_id]['hour']
        m = doc_id_to_time[doc_id]['minute']
        s = doc_id_to_time[doc_id]['second']
        new_list.append((item,year,month,day,h,m,s))
    new_list = sorted(new_list, key=itemgetter(1,2,3,4,5,6))
    return new_list
def parse_util(s):
    res = re.findall(r'\<.*?\>', s) 
    for i in range(len(res)):
        res[i] = res[i].replace('<','').replace('>','')
    return res

def prepare_input(tempfile,event_type_dict,entity_type_dict,role_type_dict,tokenizer,maxl):

    with open(tempfile) as f:
        entity = json.load(f)
    
    doc_id_to_time_json = 'doc_id_to_time.json'
    
    entity = [item[0] for item in sort_by_time(entity,doc_id_to_time_json)]
    # new_list = sort_by_time(entity,doc_id_to_time_json)

    # print(new_list[:10])
    # with open('test_order.json','w') as output:
    #     json.dump(entity, output, indent = 4, sort_keys = False) 
    # quit()
    # count = 0
    dataset = []
    labelset = []
    
    #test:
    # test_dict = {}
    # test_event = 'EndPosition'
    
    # print(tokenizer.convert_ids_to_tokens(100))
    for i in range(len(entity)-1):
        # replace_key = ''
        # for word in key.split(' '):
        #     replace_key = replace_key + word.capitalize() + ' '
        # replace_key = replace_key.strip()
        # replace_key eg. Jessica Lynch
        
        this_id = entity[i]['SENT_ID']
        next_id = entity[i+1]['SENT_ID']
        this_doc_id = entity[i]['DOC_ID']
        next_doc_id = entity[i+1]['DOC_ID']
        this_sent_idx = int(this_id.replace(this_doc_id+'-',''))
        next_sent_idx = int(next_id.replace(next_doc_id+'-','')) 
        # print(this_sent_idx,next_sent_idx)
        
        if this_doc_id == next_doc_id and abs(next_sent_idx-this_sent_idx) <= RESTRICTION_SPAN:
        # if True:
            # print(abs(next_sent_idx-this_sent_idx))
            # try get rid of <>
            first_sentence_instance = entity[i]['INSTANCE_LEVEL']
            first_sentence_role = entity[i]['ROLE_TYPE_LEVEL']
            first_sentence_entity = entity[i]['ENTITY_TYPE_LEVEL']
            instances = parse_util(first_sentence_instance)
            role_types = parse_util(first_sentence_role)
            entity_types = parse_util(first_sentence_entity)

            role_types_ids = [role_type_dict[r] for r in role_types]

            entity_types_ids = []
            for j in entity_types:
                if j not in entity_type_dict:
                    entity_types_ids.append(entity_type_dict['UNK']) # if this slot is not filled with a certain entity, thus has noe entity type level infomation
                else:
                    entity_types_ids.append(entity_type_dict[j])
            
            assert len(instances)==len(role_types) and len(instances)==len(entity_types)

            if entity[i+1]['EVENT_SUBTYPE'].strip() not in event_type_dict:
                label = event_type_dict['Null']
            else:
                label = event_type_dict[entity[i+1]['EVENT_SUBTYPE'].strip()]


            first_sentence_instance = first_sentence_instance.replace('<','').replace('>','').strip()

            #test
            # if entity[str(i)]['EVENT_SUBTYPE'] == test_event:
            #     label_type = idx_to_event[label]
            #     if label_type in test_dict:
            #         test_dict[label_type]+=1
            #     else:
            #         test_dict[label_type]=1
            #end test
            slots_index = []
            start_index = 0
            
            for ins in instances:
                start_index = first_sentence_instance.find(ins,start_index)
                slots_index.append(start_index)
            # print(slots_index)
        
            instance_token_nums = []
            to_be_replce_ids = []
            for ins in instances:
                tokenized_ins_part = tokenizer(ins,return_tensors='pt')['input_ids'][0][1:-1]
                # print(ins,tokenized_ins_part)
                instance_token_nums.append(len(tokenized_ins_part))
                to_be_replce_ids.append([t for t in tokenized_ins_part.numpy()])               
            tokenized = tokenizer(first_sentence_instance,return_tensors='pt',max_length = maxl ,padding= 'max_length')
            input_ids = tokenized['input_ids'][0]
            attention_masks = tokenized['attention_mask'][0]
            # print('input sentence tokens:', input_ids)
            # print(tokenizer.convert_ids_to_tokens(input_ids.numpy()))
            # print('instance token offsets:',instance_token_nums)
            # print('label:',label)
            # print('role ids:',role_types_ids)
            # print('entity ids:',entity_types_ids)
            replace_ids_role_type = []
            replace_ids_entity_type = []
            for i in range(len(instance_token_nums)):
                role_t = [role_types_ids[i] for _ in range(instance_token_nums[i])]
                entity_t = [entity_types_ids[i] for _ in range(instance_token_nums[i])]
                replace_ids_role_type.append(role_t)
                replace_ids_entity_type.append(entity_t)
            
            to_be_replce_ids = [item for sublist in to_be_replce_ids for item in sublist]
            replace_ids_role_type = [item for sublist in replace_ids_role_type for item in sublist]
            replace_ids_entity_type = [item for sublist in replace_ids_entity_type for item in sublist]
            # print('to_be_replce_ids:',to_be_replce_ids)
            # print('replace_ids_role_type:',replace_ids_role_type)
            # print('replace_ids_role_type:',replace_ids_entity_type)
            
            role_type_ids_tensor = input_ids.clone()
            entity_type_ids_tensor = input_ids.clone()
            
            for i in range(len(input_ids)):
                if len(to_be_replce_ids) != 0:
                    if input_ids[i] == 101:
                        role_type_ids_tensor[i] = role_type_dict['CLS']
                        entity_type_ids_tensor[i] = entity_type_dict['CLS']
                    elif input_ids[i] == 102:
                        role_type_ids_tensor[i] = role_type_dict['SEP']
                        entity_type_ids_tensor[i] = entity_type_dict['SEP']
                    elif input_ids[i] == 0:
                        role_type_ids_tensor[i] = role_type_dict['PAD']
                        entity_type_ids_tensor[i] = entity_type_dict['PAD']
                    elif input_ids[i] != to_be_replce_ids[0]:
                        role_type_ids_tensor[i] = role_type_dict['Other']
                        entity_type_ids_tensor[i] = entity_type_dict['OTHER']
                    else:
                        role_type_ids_tensor[i] = replace_ids_role_type[0]
                        entity_type_ids_tensor[i] = replace_ids_entity_type[0]
                        to_be_replce_ids = to_be_replce_ids[1:]
                        replace_ids_role_type = replace_ids_role_type[1:]
                        replace_ids_entity_type = replace_ids_entity_type[1:]
                else:
                    if input_ids[i] == 101:
                        role_type_ids_tensor[i] = role_type_dict['CLS']
                        entity_type_ids_tensor[i] = entity_type_dict['CLS']
                    elif input_ids[i] == 102:
                        role_type_ids_tensor[i] = role_type_dict['SEP']
                        entity_type_ids_tensor[i] = entity_type_dict['SEP']
                    elif input_ids[i] == 0:
                        role_type_ids_tensor[i] = role_type_dict['PAD']
                        entity_type_ids_tensor[i] = entity_type_dict['PAD']
                    else:
                        role_type_ids_tensor[i] = role_type_dict['Other']
                        entity_type_ids_tensor[i] = entity_type_dict['OTHER']
            # print('role input tensor:',role_type_ids_tensor)
            # print('entity input tensor:',entity_type_ids_tensor)

            final_data_input_tuple = (input_ids,attention_masks,role_type_ids_tensor,entity_type_ids_tensor)
            dataset.append(final_data_input_tuple)
            labelset.append(torch.tensor([label]))

    final_input_ids = []
    final_attention_masks = []
    final_role_type_ids = []
    final_entity_type_ids = []

    for tuples in dataset:  
        final_input_ids.append(tuples[0])              
        final_attention_masks.append(tuples[1])              
        final_role_type_ids.append(tuples[2])              
        final_entity_type_ids.append(tuples[3])  
                
    final_input_ids = torch.stack(final_input_ids)
    final_attention_masks = torch.stack(final_attention_masks)
    final_role_type_ids = torch.stack(final_role_type_ids)
    final_entity_type_ids = torch.stack(final_entity_type_ids)
    final_labels = torch.stack(labelset)

    # print(len(final_input_ids[0]))
    # print(len(final_attention_masks[0]))
    # print(len(final_role_type_ids[0]))
    # print(len(final_entity_type_ids[0]))
    # print(final_labels)
    # quit()

    # ordered = collections.Counter(test_dict).most_common()
    # print(test_event,':',ordered)
    # quit()

    for eg in range(1):

        print('=============input token_id example:================')
        print(dataset[eg])

        print('=============label token_id example:================')
        print(labelset[eg])


    # print(dataset['grigory pasko'])
    # quit()
    return final_input_ids,final_attention_masks,final_role_type_ids,final_entity_type_ids,final_labels

def prepare_input_withIBO(tempfile,event_type_dict,entity_type_dict,role_type_dict,tokenizer,maxl):

    with open(tempfile) as f:
        entity = json.load(f)
    
    doc_id_to_time_json = 'doc_id_to_time.json'
    
    entity = [item[0] for item in sort_by_time(entity,doc_id_to_time_json)]
    # new_list = sort_by_time(entity,doc_id_to_time_json)

    # print(new_list[:10])
    # with open('test_order.json','w') as output:
    #     json.dump(entity, output, indent = 4, sort_keys = False) 
    # quit()
    # count = 0
    dataset = []
    labelset = []
    
    #test:
    # test_dict = {}
    # test_event = 'EndPosition'
    
    # print(tokenizer.convert_ids_to_tokens(100))
    for i in range(len(entity)-1):
        event_idx = i
        # replace_key = ''
        # for word in key.split(' '):
        #     replace_key = replace_key + word.capitalize() + ' '
        # replace_key = replace_key.strip()
        # replace_key eg. Jessica Lynch
        
        this_id = entity[i]['SENT_ID']
        next_id = entity[i+1]['SENT_ID']
        this_doc_id = entity[i]['DOC_ID']
        next_doc_id = entity[i+1]['DOC_ID']
        this_sent_idx = int(this_id.replace(this_doc_id+'-',''))
        next_sent_idx = int(next_id.replace(next_doc_id+'-','')) 
        # print(this_sent_idx,next_sent_idx)
        
        if this_doc_id == next_doc_id and abs(next_sent_idx-this_sent_idx) <= RESTRICTION_SPAN:
        # if True:
            # print(abs(next_sent_idx-this_sent_idx))
            # try get rid of <>
            first_sentence_instance = entity[i]['INSTANCE_LEVEL']
            first_sentence_role = entity[i]['ROLE_TYPE_LEVEL']
            first_sentence_entity = entity[i]['ENTITY_TYPE_LEVEL']
            instances = parse_util(first_sentence_instance)
            role_types = parse_util(first_sentence_role)
            entity_types = parse_util(first_sentence_entity)

            role_types_ids = [(role_type_dict['B-'+r],role_type_dict['I-'+r]) for r in role_types]

            entity_types_ids = []
            for j in entity_types:
                jj = 'B-'+j
                if jj not in entity_type_dict:
                    entity_types_ids.append((entity_type_dict['B-'+'UNK'],entity_type_dict['I-'+'UNK'])) # if this slot is not filled with a certain entity, thus has noe entity type level infomation
                else:
                    entity_types_ids.append((entity_type_dict['B-'+j],entity_type_dict['I-'+j]))
            
            assert len(instances)==len(role_types) and len(instances)==len(entity_types)

            if entity[i+1]['EVENT_SUBTYPE'].strip() not in event_type_dict:
                label = event_type_dict['Null']
            else:
                label = event_type_dict[entity[i+1]['EVENT_SUBTYPE'].strip()]


            first_sentence_instance = first_sentence_instance.replace('<','').replace('>','').strip()

            #test
            # if entity[str(i)]['EVENT_SUBTYPE'] == test_event:
            #     label_type = idx_to_event[label]
            #     if label_type in test_dict:
            #         test_dict[label_type]+=1
            #     else:
            #         test_dict[label_type]=1
            #end test
            slots_index = []
            start_index = 0
            
            for ins in instances:
                start_index = first_sentence_instance.find(ins,start_index)
                slots_index.append(start_index)
            # print(slots_index)
        
            instance_token_nums = []
            to_be_replce_ids = []
            for ins in instances:
                tokenized_ins_part = tokenizer(ins,return_tensors='pt')['input_ids'][0][1:-1]
                # print(ins,tokenized_ins_part)
                instance_token_nums.append(len(tokenized_ins_part))
                to_be_replce_ids.append([t for t in tokenized_ins_part.numpy()])               
            tokenized = tokenizer(first_sentence_instance,return_tensors='pt',max_length = maxl ,padding= 'max_length')
            
            
            input_ids = tokenized['input_ids'][0]
            
            
            attention_masks = tokenized['attention_mask'][0]
            # print('input sentence tokens:', input_ids)
            # print(tokenizer.convert_ids_to_tokens(input_ids.numpy()))
            # print('instance token offsets:',instance_token_nums)
            # print('label:',label)
            # print('role ids:',role_types_ids)
            # print('entity ids:',entity_types_ids)
            replace_ids_role_type = []
            replace_ids_entity_type = []
            for i in range(len(instance_token_nums)):
                role_t = []
                entity_t = []
                for j in range(instance_token_nums[i]):
                    if j == 0:
                        role_t.append(role_types_ids[i][0]) 
                        entity_t.append(entity_types_ids[i][0])
                    else:
                        role_t.append(role_types_ids[i][1]) 
                        entity_t.append(entity_types_ids[i][1])

                replace_ids_role_type.append(role_t)
                replace_ids_entity_type.append(entity_t)
            
            to_be_replce_ids = [item for sublist in to_be_replce_ids for item in sublist]
            replace_ids_role_type = [item for sublist in replace_ids_role_type for item in sublist]
            replace_ids_entity_type = [item for sublist in replace_ids_entity_type for item in sublist]
            # print('to_be_replce_ids:',to_be_replce_ids)
            # print('replace_ids_role_type:',replace_ids_role_type)
            # print('replace_ids_role_type:',replace_ids_entity_type)
            if event_idx ==3:
                print('')
                print(tokenizer.convert_ids_to_tokens(input_ids.numpy()))
                print(role_types_ids)
                print(entity_types_ids)
                print(to_be_replce_ids)
                print(replace_ids_role_type)
                print(replace_ids_entity_type)

            role_type_ids_tensor = input_ids.clone()
            entity_type_ids_tensor = input_ids.clone()
            
            for i in range(len(input_ids)):
                if len(to_be_replce_ids) != 0:
                    if input_ids[i] == 101:
                        role_type_ids_tensor[i] = role_type_dict['CLS']
                        entity_type_ids_tensor[i] = entity_type_dict['CLS']
                    elif input_ids[i] == 102:
                        role_type_ids_tensor[i] = role_type_dict['SEP']
                        entity_type_ids_tensor[i] = entity_type_dict['SEP']
                    elif input_ids[i] == 0:
                        role_type_ids_tensor[i] = role_type_dict['PAD']
                        entity_type_ids_tensor[i] = entity_type_dict['PAD']
                    elif input_ids[i] != to_be_replce_ids[0]:
                        role_type_ids_tensor[i] = role_type_dict['Other']
                        entity_type_ids_tensor[i] = entity_type_dict['OTHER']
                    else:
                        role_type_ids_tensor[i] = replace_ids_role_type[0]
                        entity_type_ids_tensor[i] = replace_ids_entity_type[0]
                        to_be_replce_ids = to_be_replce_ids[1:]
                        replace_ids_role_type = replace_ids_role_type[1:]
                        replace_ids_entity_type = replace_ids_entity_type[1:]
                else:
                    if input_ids[i] == 101:
                        role_type_ids_tensor[i] = role_type_dict['CLS']
                        entity_type_ids_tensor[i] = entity_type_dict['CLS']
                    elif input_ids[i] == 102:
                        role_type_ids_tensor[i] = role_type_dict['SEP']
                        entity_type_ids_tensor[i] = entity_type_dict['SEP']
                    elif input_ids[i] == 0:
                        role_type_ids_tensor[i] = role_type_dict['PAD']
                        entity_type_ids_tensor[i] = entity_type_dict['PAD']
                    else:
                        role_type_ids_tensor[i] = role_type_dict['Other']
                        entity_type_ids_tensor[i] = entity_type_dict['OTHER']
            if event_idx ==3:
                print('')
                print('role input tensor:',role_type_ids_tensor)
                print('entity input tensor:',entity_type_ids_tensor)

            final_data_input_tuple = (input_ids,attention_masks,role_type_ids_tensor,entity_type_ids_tensor)
            dataset.append(final_data_input_tuple)
            labelset.append(torch.tensor([label]))

    final_input_ids = []
    final_attention_masks = []
    final_role_type_ids = []
    final_entity_type_ids = []

    for tuples in dataset:  
        final_input_ids.append(tuples[0])              
        final_attention_masks.append(tuples[1])              
        final_role_type_ids.append(tuples[2])              
        final_entity_type_ids.append(tuples[3])  
                
    final_input_ids = torch.stack(final_input_ids)
    final_attention_masks = torch.stack(final_attention_masks)
    final_role_type_ids = torch.stack(final_role_type_ids)
    final_entity_type_ids = torch.stack(final_entity_type_ids)
    final_labels = torch.stack(labelset)

    # print(len(final_input_ids[0]))
    # print(len(final_attention_masks[0]))
    # print(len(final_role_type_ids[0]))
    # print(len(final_entity_type_ids[0]))
    # print(final_labels)
    # quit()

    # ordered = collections.Counter(test_dict).most_common()
    # print(test_event,':',ordered)
    # quit()

    # eg = 3
    # print('=============input token_id example:================')
    # print(dataset[eg])

    # print('=============label token_id example:================')
    # print(labelset[eg])


    # print(dataset['grigory pasko'])
    # quit()
    return final_input_ids,final_attention_masks,final_role_type_ids,final_entity_type_ids,final_labels
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# word = 'Michael Crichton'
# ids = tokenizer(word,return_tensors='pt')['input_ids']
# print(ids)
# print(tokenizer.convert_ids_to_tokens(ids[0].tolist()))

# prepare_input('ACE05_events_three_level_train_with_sent_id.json',event_type_dict,entity_type_dict,role_type_dict,BertTokenizer.from_pretrained('bert-base-cased'),200)
