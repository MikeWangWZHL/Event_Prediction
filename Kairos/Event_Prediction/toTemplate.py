import json

'''create json file from txt'''
# filename = 'all_files.txt'
# all_centroid_entity = {}
# current_entity = ""
# event_count = 0
# with open(filename) as f:
#     for line in f:
#         striped_line = line.strip()
#         if striped_line == '':
#             continue
#         # print(f'-{striped_line}-')
#         if striped_line.islower():
#             print(striped_line)
#             current_entity = striped_line
#             event_count = 0
#             continue
#         else:
#             # print(striped_line)
#             key, value = striped_line.split(maxsplit=1)
#             if key == 'DOCID':
#                 event_count += 1
#             if current_entity not in all_centroid_entity:
#                 all_centroid_entity[current_entity] = {}
#             if event_count not in all_centroid_entity[current_entity]:
#                 all_centroid_entity[current_entity][event_count] = {}
#             all_centroid_entity[current_entity][event_count][key] = value.strip()

# out_file = open("test1.json", "w") 
# json.dump(all_centroid_entity, out_file, indent = 4, sort_keys = False) 
# out_file.close() 




## comparing subtypes
# event_set = set()

# with open('test1.json') as jf:
#     data = json.load(jf)
#     for _,events in data.items():
#         for _,event in events.items():
#             event_set.add(event['EVENT_SUBTYPE'])

# # print(event_set)
# event_set_trim = set()
# for event in event_set:
#     event_set_trim.add(event.replace('-',''))
# # print(event_set_trim)


# aida_kairos_event_set = set()
# with open('aida_events.json') as f:
#     aida_data = json.load(f)
#     for event in aida_data:
#         aida_kairos_event_set.add(event['Subtype'])

# with open('kairos_events.json') as f:
#     kairos_data = json.load(f)
#     for event in kairos_data:
#         aida_kairos_event_set.add(event['Subtype'])

# # print(aida_kairos_event_set)
# ifHasTemplate = []
# for event in event_set_trim:
#     if event in aida_kairos_event_set:
#         ifHasTemplate.append((event,True))

#     else:
#         ifHasTemplate.append((event,False))
# print(ifHasTemplate)
        

# start converting:
temps = {}
with open('ace_templates.json') as f:
    temps = json.load(f)

exist_event_set = set()

with open('all_data.json') as jf:
    data = json.load(jf)
    temp_dict = data.copy()
    for entity,events in data.items():
        for key,event in events.items():
            event_type = event['EVENT_SUBTYPE'].strip().replace('-','')
            exist_event_set.add(event_type)
            if event_type in temps:
                temp_str_instance_level = temps[event_type]
                temp_str_entitytype_level = temps[event_type]

                for arg,value in event.items():
                    entity_type = arg+'_TYPE'
                    arg = arg.capitalize()
                    if arg in temp_str_instance_level:
                        temp_str_instance_level = temp_str_instance_level.replace(arg,value)
                        if entity_type in event:
                            temp_str_entitytype_level = temp_str_entitytype_level.replace(arg,event[entity_type])
                
                temp_dict[entity][key] = {'EVENT_SUBTYPE':event_type, 'INSTANCE_LEVEL':temp_str_instance_level, 'ROLE_TYPE_LEVEL':temps[event_type],'ENTITY_TYPE_LEVEL':temp_str_entitytype_level}


# print(exist_event_set)
for e in exist_event_set:
    print(f'{e}:    {temps[e]}')
out_file = open("test_temp_three_level.json", "w") 
json.dump(temp_dict, out_file, indent = 4, sort_keys = False) 
out_file.close() 