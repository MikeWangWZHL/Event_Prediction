from excel2json import convert_from_file
import json

# excel_file = 'AIDA_Annotation_Ontology_Phase2_V1.xlsx'
# convert_from_file(excel_file)
# with open('aida.txt','w+') as output:
#     json.dump(excel_file,output)


# aida_file = './aida_data/events.json'
# kairos_file = './kairos_data/events.json'
aida_file = './aida_data/relations.json'
kairos_file = './kairos_data/relations.json'

with open(aida_file,'r') as readfile:
    aida_data = json.load(readfile)
with open(kairos_file,'r') as readfile2:
    kairos_data = json.load(readfile2)

is_fine_grain = True

events = {}
if is_fine_grain:
    # add AIDA
    for item in aida_data:
        label = item['Type'] + '.' + item['Subtype']
        if item['Sub-subtype'] == 'n/a':
            label += '.<n/a>'
        else:
            label = label + '.' + item['Sub-subtype']
        
        slots = {}
        template_str = item['Template'].split(' ')
        arg_count = 0
        for word in template_str:
            if '<arg' in word:
                arg_count += 1
        order = []
        for i in range(1,arg_count+1):
            arg_label = 'arg' + str(i) + ' label'
            arg_constrains = 'arg' + str(i) + ' type constraints'
            slots[item[arg_label].strip()] = [ c for c in item[arg_constrains].split(', ')]
            order.append(item[arg_label].strip())
        
        event = {}
        event['label'] = label
        event['template'] = item['Template']
        event['arg_order'] = order 
        event['slots'] = slots
        event['definition'] = item['Definition']
        event['origin'] = 'AIDA'

        events[label] = event
        print("adia: ",label)
        # if label == 'Conflict.Attack':
        #     print(slots)
        # print(events)
        # print(label)
        # print(slots)
        # print('====================================================================')

    # add kairos
    for item in kairos_data:
        label = item['Type'] + '.' + item['Subtype']
        if item['Sub-subtype'] == 'Unspecified':
            label += '.<n/a>'
        else:
            label = label + '.' + item['Sub-subtype']

        kairos_slots = {}
        template_str = item['Template'].split(' ')
        arg_count = 0
        for word in template_str:
            if 'arg' in word:
                arg_count += 1

        kairos_order = []
        for i in range(1,arg_count+1):
            arg_label = 'arg' + str(i) + ' label'
            arg_constrains = 'arg' + str(i) + ' type constraints'
            kairos_slots[item[arg_label].strip()] = [ c for c in item[arg_constrains].split(', ')]
            kairos_order.append(item[arg_label].strip())

        if label in events:
            event = events[label]
            event['arg_order_aida'] = event['arg_order'].copy()
            event['Aida_slots'] = event['slots']            
            event['arg_order_kairos'] = kairos_order
            event['Kairos_slots'] = kairos_slots
            event['origin'] = 'AIDA/KAIROS'
            event['arg_order'] = []

            merged_slots = {}
            for key,value in kairos_slots.items():
                if key in merged_slots:
                    merged_slots[key] = list(set().union(value,event['slots'][key]))
                else:
                    merged_slots[key] = value
            
            event['slots'] = merged_slots
            if label == "Contact.Prevarication.Broadcast":
                # print(event['slots'])
                # print(event['Aida_slots'])
                # print(event['Kairos_slots'])
                print(event)
            if not item['Definition'] == event['definition']:
                event['definition'] = event['definition'] + '||' + item['Definition']
                # print(event['definition'])
            if not item['Template'] == event['template']:
                event['template'] = event['template'] + '||' + item['Template']
                # print(event['template'])
            print("aida/kairos",label)
            # print("kairos: ", kairos_slots)
            # print("aida: ",event['Aida_slots'])
            # print('merged: ',event['slots'])
            # print('====================================================================')
        else:
            event = {}
            event['label'] = label
            event['template'] = item['Template']
            event['arg_order'] = kairos_order
            event['slots'] = kairos_slots
            event['definition'] = item['Definition']
            event['origin'] = 'KAIROS'
            events[label] = event
            print("Kairos: ",label)
else:
    # add AIDA
    for item in aida_data:
        label = item['Type'] + '.' + item['Subtype']
        if label in events:
            if item['Sub-subtype'] != 'n/a':
                continue
        
        slots = {}
        template_str = item['Template'].split(' ')
        arg_count = 0
        for word in template_str:
            if 'arg' in word:
                arg_count += 1
        for i in range(1,arg_count+1):
            arg_label = 'arg' + str(i) + ' label'
            arg_constrains = 'arg' + str(i) + ' type constraints'
            slots[item[arg_label].strip()] = [ c for c in item[arg_constrains].split(', ')]

        event = {}
        event['label'] = label
        event['slots'] = slots
        event['definition'] = item['Definition']
        event['template'] = item['Template']
        event['origin'] = 'AIDA'

        events[label] = event
        # if label == 'Conflict.Attack':
        #     print(slots)
        # print(events)
        # print(label)
        # print(slots)
        # print('====================================================================')

    # add kairos
    for item in kairos_data:
        label = item['Type'] + '.' + item['Subtype']
        if item['Sub-subtype'] != 'Unspecified':
            continue

        kairos_slots = {}
        template_str = item['Template'].split(' ')
        arg_count = 0
        for word in template_str:
            if 'arg' in word:
                arg_count += 1
        for i in range(1,arg_count+1):
            arg_label = 'arg' + str(i) + ' label'
            arg_constrains = 'arg' + str(i) + ' type constraints'
            kairos_slots[item[arg_label].strip()] = [ c for c in item[arg_constrains].split(', ')]

        if label in events:
            event = events[label]
            event['origin'] = 'AIDA/KAIROS'
            event['Aida_slots'] = event['slots']
            event['Kairos_slots'] = kairos_slots
            merged_slots = event['Aida_slots']
            for key,value in kairos_slots.items():
                if key in merged_slots:
                    merged_slots[key] = list(set().union(value,merged_slots[key]))
                else:
                    merged_slots[key] = value
            
            event['slots'] = merged_slots
            if not item['Definition'] == event['definition']:
                event['definition'] = event['definition'] + '||' + item['Definition']
                # print(event['definition'])
            if not item['Template'] == event['template']:
                event['template'] = event['template'] + '||' + item['Template']
                # print(event['template'])
            # print(label)
            # print("kairos: ", kairos_slots)
            # print("aida: ",event['Aida_slots'])
            # print('merged: ',event['slots'])
            # print('====================================================================')
        else:
            event = {}
            event['label'] = label
            event['slots'] = kairos_slots
            event['definition'] = item['Definition']
            event['template'] = item['Template']
            event['origin'] = 'KAIROS'
            events[label] = event

merged_list = []
for _,value in events.items():
    merged_list.append(value)

with open('test.json','w') as f:
    json.dump(merged_list,f,indent=4)

