from transformers import BertTokenizer, BertForMaskedLM
import torch 
import json
import re
import collections

# "EVENT_SUBTYPE": "Injure",
# "INSTANCE_LEVEL": "<jessica lynch> was injured by <Agent> using <Instrument> at <Place> place on <NaN>",
# "ROLE_TYPE_LEVEL": "<Victim> was injured by <Agent> using <Instrument> at <Place> place on <Time>",
# "ENTITY_TYPE_LEVEL": "<PER> was injured by <Agent> using <Instrument> at <Place> place on <Time>"
event_type_dict = {'Divorce': 0, 'EndPosition': 1, 'Acquit': 2, 'Meet': 3, 'Die': 4, 'Extradite': 5, 'Sue': 6, 'Elect': 7, 'Convict': 8, 'TransferOwnership': 9, 'Marry': 10, 'Attack': 11, 'StartPosition': 12, 'ArrestJail': 13, 'ReleaseParole': 14, 'Nominate': 15, 'Transport': 16, 'Fine': 17, 'Sentence': 18, 'TrialHearing': 19, 'BeBorn': 20, 'Pardon': 21, 'Demonstrate': 22, 'Execute': 23, 'StartOrg': 24, 'PhoneWrite': 25, 'Appeal': 26, 'Injure': 27, 'ChargeIndict': 28, 'TransferMoney': 29, 'Null':30}

entity_type_dict = {'ORG': 0, 'LOC': 1, 'VEH': 2, 'WEA': 3, 'GPE': 4, 'FAC': 5, 'PER': 6, 'CLS':7, 'SEP':8, 'PAD':9, 'UNK':10}
for e in entity_type_dict.keys():
    entity_type_dict[e] += 1
role_type_dict = {'Target': 0, 'Plaintiff': 1, 'Person': 2, 'Seller': 3, 'Time': 4, 'Recipient': 5, 'Instrument': 6, 'Artifact': 7, 'Adjudicator': 8, 'Prosecutor': 9, 'Agent': 10, 'Beneficiary': 11, 'Attacker': 12, 'Victim': 13, 'Money': 14, 'Buyer': 15, 'Docid': 16, 'Crime': 17, 'Giver': 18, 'Sentence': 19, 'Org': 20, 'Defendant': 21, 'Position': 22, 'Vehicle': 23, 'Destination': 24, 'Origin': 25, 'Place': 26, 'Entity': 27, 'Price':28, 'CLS':29, 'SEP':30, 'PAD':31} 
for r in role_type_dict.keys():
    role_type_dict[r] += 1
entity_type_dict['OTHER'] = 0
role_type_dict['Other'] = 0

idx_to_event = {value:key for key,value in event_type_dict.items()}
idx_to_role = {value:key for key,value in role_type_dict.items()}
idx_to_entity = {value:key for key,value in entity_type_dict.items()}


def parse_util(s):
    res = re.findall(r'\<.*?\>', s) 
    for i in range(len(res)):
        res[i] = res[i].replace('<','').replace('>','')
    return res

def prepare_input(tempfile,event_type_dict,entity_type_dict,role_type_dict,tokenizer):

    with open(tempfile) as f:
        temps = json.load(f)

    # count = 0
    dataset = []
    labelset = []
    
    #test:
    test_dict = {}
    test_event = 'EndPosition'
    test_event = 'Attack'
    test_event = 'Die'
    test_event = 'StartPosition'


    # print(tokenizer.convert_ids_to_tokens(100))
    for key, entity in temps.items():
        train_length = len(entity)
        if train_length >= 2:
            for i in range(1,train_length):

                # replace_key = ''
                # for word in key.split(' '):
                #     replace_key = replace_key + word.capitalize() + ' '
                # replace_key = replace_key.strip()
                # replace_key eg. Jessica Lynch
                

                # try get rid of <>
                first_sentence_instance = entity[str(i)]['INSTANCE_LEVEL']
                first_sentence_role = entity[str(i)]['ROLE_TYPE_LEVEL']
                first_sentence_entity = entity[str(i)]['ENTITY_TYPE_LEVEL']
                instances = parse_util(first_sentence_instance)
                role_types = parse_util(first_sentence_role)
                entity_types = parse_util(first_sentence_entity)

                role_types_ids = [role_type_dict[r] for r in role_types]

                entity_types_ids = []
                for j in entity_types:
                    if j not in entity_type_dict:
                        entity_types_ids.append(0) # if this slot is not filled with a certain entity, thus has noe entity type level infomation
                    else:
                        entity_types_ids.append(entity_type_dict[j])
                
                assert len(instances)==len(role_types) and len(instances)==len(entity_types)

                if entity[str(i+1)]['EVENT_SUBTYPE'].strip() not in event_type_dict:
                    label = event_type_dict['Null']
                else:
                    label = event_type_dict[entity[str(i+1)]['EVENT_SUBTYPE'].strip()]


                first_sentence_instance = first_sentence_instance.replace('<','').replace('>','').strip()

                #test
                if entity[str(i)]['EVENT_SUBTYPE'] == test_event:
                    label_type = idx_to_event[label]
                    if label_type in test_dict:
                        test_dict[label_type]+=1
                    else:
                        test_dict[label_type]=1
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
                tokenized = tokenizer(first_sentence_instance,return_tensors='pt',max_length = 64 ,padding= 'max_length')
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

                    if input_ids[i] == 101:
                        role_type_ids_tensor[i] = 0
                        entity_type_ids_tensor[i] = 0
                    elif input_ids[i] == 102:
                        role_type_ids_tensor[i] = 0
                        entity_type_ids_tensor[i] = 0
                    elif input_ids[i] == 0:
                        role_type_ids_tensor[i] = 0
                        entity_type_ids_tensor[i] = 0
                    elif input_ids[i] != to_be_replce_ids[0]:
                        role_type_ids_tensor[i] = role_type_dict['Other']
                        entity_type_ids_tensor[i] = entity_type_dict['OTHER']
                    else:
                        role_type_ids_tensor[i] = replace_ids_role_type[0]
                        entity_type_ids_tensor[i] = replace_ids_entity_type[0]
                        to_be_replce_ids = to_be_replce_ids[1:]
                        replace_ids_role_type = replace_ids_role_type[1:]
                        replace_ids_entity_type = replace_ids_entity_type[1:]

                # print('role input tensor:',role_type_ids_tensor)
                # print('entity input tensor:',entity_type_ids_tensor)



                final_data_input_tuple = (input_ids,attention_masks,role_type_ids_tensor,entity_type_ids_tensor)
                dataset.append(final_data_input_tuple)
                labelset.append(torch.tensor([label]))
                # print(tokenizer.convert_ids_to_tokens(input_ids.numpy()))
                # print(dataset)
                # print(labelset)
                # quit()
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

    ordered = collections.Counter(test_dict).most_common()
    print(test_event,':',ordered)
    # quit()

    for eg in range(5):

        print('=============input token_id example:================')
        print(dataset[eg])

        print('=============label token_id example:================')
        print(labelset[eg])


    # print(dataset['grigory pasko'])
    # quit()
    return final_input_ids,final_attention_masks,final_role_type_ids,final_entity_type_ids,final_labels

# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# word = 'Michael Crichton'
# ids = tokenizer(word,return_tensors='pt')['input_ids']
# print(ids)
# print(tokenizer.convert_ids_to_tokens(ids[0].tolist()))
# prepare_input('test_temp_three_level.json',event_type_dict,entity_type_dict,role_type_dict,BertTokenizer.from_pretrained('bert-base-cased'))