import json

# # generate aida templates
with open('aida_events.json') as f:
    data = json.load(f)
    is_added = {}
    for event in data:
        # if event['Subtype'] in is_added: 
        #     continue
        # else:
        label = event['Type']+'.'+event['Subtype'] + '.' + event['Sub-subtype']
        template_str = event['Template']
        template_str_splited = template_str.strip().split(' ')
        for word in template_str_splited:
            if '<' in word:
                arg_str = word[1:5]
                key = arg_str + ' label'
                replace_token = '<'+arg_str+'>'
                # print(replace_token)
                new_token = '<'+event[key].strip()+'>'
                # print(new_token)
                template_str = template_str.replace(replace_token,new_token)
        # print(template_str)

        # is_added[event['Subtype']] = template_str
        # print(key)
        is_added[label] = template_str

out_file = open("aida_templates_whole.json", "w") 
json.dump(is_added, out_file, indent = 4, sort_keys = False) 
out_file.close() 



# # generate kairos templates
# with open('kairos_events.json') as f:
#     data = json.load(f)
#     is_added = {}
#     for event in data:
#         if event['Sub-subtype'] != "Unspecified": 
#             continue
#         else:
#             template_str = event['Template']
#             template_str_splited = template_str.strip().split(' ')
#             for word in template_str_splited:
#                 if '<' in word:
#                     arg_str = word[1:5]
#                     key = arg_str + ' label'
#                     replace_token = '<'+arg_str+'>'
#                     # print(replace_token)
#                     new_token = '<'+event[key].strip()+'>'
#                     # print(new_token)
#                     template_str = template_str.replace(replace_token,new_token)
#             print(template_str)

#             is_added[event['Subtype']] = template_str

# out_file = open("kairos_templates.json", "w") 
# json.dump(is_added, out_file, indent = 4, sort_keys = False) 
# out_file.close() 

