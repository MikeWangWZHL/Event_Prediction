import json
from nltk.tokenize.treebank import TreebankWordDetokenizer

data = []
with open('train.oneie.json') as f:
    for line in f:
        line_item = json.loads(line)
        if line_item['event_mentions'] != []:
            data.append(line_item)

event_type_set = set()
entity_type_set = set()
role_type_set = set()

# see the whole data as a sequenced dataset
ACE05_events_three_level = []
for raw_item in data:

    # print('sentence:',raw_item['sentence'],'\n')
    # print('tokens:',raw_item['tokens'],'\n')
    # print('entity_mentions:',raw_item['entity_mentions'],'\n')
    # print('event_mentions:',raw_item['event_mentions'],'\n')
    # print('')

    id2entity = {}
    for entity in raw_item['entity_mentions']:
        id2entity[entity['id']] = (entity['start'],entity['end'],entity['entity_type'])
    # print(id2entity,'\n')
    #(start_token_position,end_token_position,entity_type)


    raw_sentence = raw_item['sentence']
    raw_tokens = raw_item['tokens']
    sentence_id = raw_item['sent_id']
    # time = 
    doc_id = raw_item['doc_id']
    for e in raw_item['event_mentions']:
        ouput_item = {}
        
        e_type = e['event_type']  
        args = e['arguments']
        ouput_item['EVENT_SUBTYPE'] = e_type.split(':')[1].replace('-','').strip()
        ouput_item['SENT_ID'] = sentence_id
        ouput_item['DOC_ID'] = doc_id
        ouput_item['EVENT_ID'] = e['id']
        event_type_set.add(e_type.split(':')[1].replace('-','').strip())
        
        instance_level_tokens = raw_tokens.copy()
        entity_level_tokens = raw_tokens.copy()
        role_level_tokens = raw_tokens.copy()

        for arg in args:
            id = arg['entity_id']
            role_level = arg['role']
            instance_level = arg['text']
            start_position = id2entity[id][0]
            end_position = id2entity[id][1]
            entity_level = id2entity[id][2]

            entity_type_set.add(entity_level)
            role_type_set.add(role_level)

            del instance_level_tokens[start_position:end_position]
            instance_level_tokens.insert(start_position,'<'+instance_level+'>')
            del entity_level_tokens[start_position:end_position]
            entity_level_tokens.insert(start_position,'<'+entity_level+'>')
            del role_level_tokens[start_position:end_position]
            role_level_tokens.insert(start_position,'<'+role_level+'>')
        ouput_item['INSTANCE_LEVEL'] = TreebankWordDetokenizer().detokenize(instance_level_tokens)
        ouput_item['ROLE_TYPE_LEVEL'] = TreebankWordDetokenizer().detokenize(role_level_tokens)
        ouput_item['ENTITY_TYPE_LEVEL'] = TreebankWordDetokenizer().detokenize(entity_level_tokens)
        
        ACE05_events_three_level.append(ouput_item)

    # print('====================')
    # print(ouput_item)
print(event_type_set)
print(entity_type_set)
print(role_type_set)

##output
with open('test_event_time_order.json','w') as out_file:
    json.dump(ACE05_events_three_level, out_file, indent = 4, sort_keys = False) 
