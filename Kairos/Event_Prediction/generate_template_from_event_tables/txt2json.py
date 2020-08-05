import json

'''create json file from txt'''
filename = 'all_files_emma_new.txt'
all_centroid_entity = {}
current_entity = ""
event_count = 0
with open(filename) as f:
    for line in f:
        striped_line = line.strip()
        if striped_line == '':
            continue
        # print(f'-{striped_line}-')
        if striped_line.islower():
            # print(striped_line)
            current_entity = striped_line
            event_count = 0
            continue
        else:
            # print(striped_line)
            key, value = striped_line.split(maxsplit=1)
            if key == 'DOCID':
                event_count += 1
            if current_entity not in all_centroid_entity:
                all_centroid_entity[current_entity] = {}
            if event_count not in all_centroid_entity[current_entity]:
                all_centroid_entity[current_entity][event_count] = {}

            if '(' in value.strip():
                
                role_type = value.strip().split('(')[1].replace(')','').strip()
                if len(role_type) == 3 and role_type.isupper():
                    role_type_key = key+'_TYPE'
                    all_centroid_entity[current_entity][event_count][role_type_key] = role_type

            all_centroid_entity[current_entity][event_count][key] = value.strip().split('(')[0].strip()

out_file = open("test1.json", "w") 
json.dump(all_centroid_entity, out_file, indent = 4, sort_keys = False) 
out_file.close()