import json


lookuptable = {}
with open('./ACE05_data/train.oneie.json') as f:
    for line in f:
        line_item = json.loads(line)
        if line_item['event_mentions'] != []:
            for em in line_item['event_mentions']:
                em_id = em['id']
                assert em_id not in lookuptable
                lookuptable[em_id] = em


with open('./ACE05_data/dev.oneie.json') as f:
    for line in f:
        line_item = json.loads(line)
        if line_item['event_mentions'] != []:
            for em in line_item['event_mentions']:
                em_id = em['id']
                assert em_id not in lookuptable
                lookuptable[em_id] = em
                


with open('./ACE05_data/test.oneie.json') as f:
    for line in f:
        line_item = json.loads(line)
        if line_item['event_mentions'] != []:
            for em in line_item['event_mentions']:
                em_id = em['id']
                assert em_id not in lookuptable
                lookuptable[em_id] = em
                
print(lookuptable['CNN_CF_20030303.1900.02-EV1-1'])
print(lookuptable['AFP_ENG_20030323.0020-EV1-2'])
print(lookuptable['AGGRESSIVEVOICEDAILY_20050107.2012-EV1-1'])
print(lookuptable['AFP_ENG_20030601.0262-EV11-1'])


with open('em_id_lookup.json','w') as out_file:
    json.dump(lookuptable, out_file, indent = 4, sort_keys = True) 
