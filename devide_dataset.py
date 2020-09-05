import json
import random
with open('event_chains_total.json') as f:
    dataset = json.load(f)
print('size:',len(dataset))
random.shuffle(dataset)

train_data = dataset[:595]
test_data = dataset[595:]
print('train size:',len(train_data))
with open('event_chains_train.json','w') as f:
    json.dump(train_data, f, indent = 4, sort_keys = False) 
print('test size:',len(test_data))
with open('event_chains_test.json','w') as f:
    json.dump(test_data, f, indent = 4, sort_keys = False) 