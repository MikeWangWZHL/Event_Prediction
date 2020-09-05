from transformers import BertTokenizer, BertConfig
from transformers import AdamW
from modeling_bert import BertForSequenceClassification
# from transformers import BertForSequenceClassification
import json
import torch 
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine
from torch import nn

# from gen_event_entity_dict import gen_event_entity_role_dict
# from prepare_input_ACE05 import prepare_input
from prepare_input_ACE05_with_sent_id import prepare_input, prepare_input_withIBO_two_pair, prepare_input_withIBO_multi_pair, prepare_input_withIBO_multi_pair_individual
from prepare_input_v2 import prepare_input_emma



if torch.cuda.is_available():  
  dev = "cuda:2" 
else:  
  dev = "cpu"
CUDA_VISIBLE_DEVICES=2  
device = torch.device(dev)

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


print(event_type_dict)
print(entity_type_dict)
print(role_type_dict)





"""set up tokenizer"""
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

"""set up model"""
model = BertForSequenceClassification.from_pretrained('./model_save_ACE05_doc_time_order_pretrain_batch16')
model.cuda(2)

def trim_batch(input_ids, pad_token_id, role_type_ids, entity_type_ids, labels, attention_mask=None):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return (input_ids[:, keep_column_mask], None,  role_type_ids[:, keep_column_mask], entity_type_ids[:, keep_column_mask], labels)
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask],  role_type_ids[:, keep_column_mask], entity_type_ids[:, keep_column_mask], labels)


"""set up input"""
tokenizer_max_len = 200
print('\n===================================')
print('prediction:')
item = ('AFP_ENG_20030305.0918-EV8-14', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1')
input_ids,attention_masks,role_type_ids,entity_type_ids,labels = prepare_input_withIBO_multi_pair_individual(item,event_type_dict,entity_type_dict,role_type_dict,tokenizer,tokenizer_max_len)
batch = trim_batch(input_ids, tokenizer.pad_token_id, role_type_ids,entity_type_ids,labels,attention_masks)
print('input_ids:\n',batch[0])
print('')


with torch.no_grad():        

    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_role_type_ids = batch[2].to(device)
    b_entity_type_ids = batch[3].to(device)
    b_labels = batch[4].to(device)
    outputs = model(b_input_ids, attention_mask=b_input_mask,role_type_ids=b_role_type_ids,entity_type_ids=b_entity_type_ids, labels=b_labels)
    # outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
    loss = outputs[0] 
    logits = outputs[1]

    # logits = logits.detach().cpu().numpy()
    logits = logits.detach().cpu()
    label_ids = b_labels.to('cpu').numpy()
print(logits)

print('label: ',idx_to_event[label_ids[0][0]])
print('top3 prediction: ')
for p in torch.topk(logits[0],3).indices.numpy():
    print('\t',idx_to_event[p])
# print(np.argmax(logits))



# [('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-11'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-14'), ('AFP_ENG_20030305.0918-EV8-1', 'AFP_ENG_20030305.0918-EV20-1'), ('AFP_ENG_20030305.0918-EV8-11', 'AFP_ENG_20030305.0918-EV20-1'), ('AFP_ENG_20030305.0918-EV8-14', 'AFP_ENG_20030305.0918-EV20-1'), ('AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1'), ('AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1'), ('AFP_ENG_20030305.0918-EV28-1', 'AFP_ENG_20030305.0918-EV12-1'), ('AFP_ENG_20030305.0918-EV28-1', 'AFP_ENG_20030305.0918-EV34-1'), ('AFP_ENG_20030305.0918-EV28-1', 'AFP_ENG_20030305.0918-EV35-1'), ('AFP_ENG_20030305.0918-EV29-1', 'AFP_ENG_20030305.0918-EV12-1'), ('AFP_ENG_20030305.0918-EV29-1', 'AFP_ENG_20030305.0918-EV34-1'), ('AFP_ENG_20030305.0918-EV29-1', 'AFP_ENG_20030305.0918-EV35-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-1', 'AFP_ENG_20030305.0918-EV20-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-11', 'AFP_ENG_20030305.0918-EV20-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-14', 'AFP_ENG_20030305.0918-EV20-1'), ('AFP_ENG_20030305.0918-EV8-1', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1'), ('AFP_ENG_20030305.0918-EV8-1', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1'), ('AFP_ENG_20030305.0918-EV8-11', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1'), ('AFP_ENG_20030305.0918-EV8-11', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1'), ('AFP_ENG_20030305.0918-EV8-14', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1'), ('AFP_ENG_20030305.0918-EV8-14', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1'), ('AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1', 'AFP_ENG_20030305.0918-EV12-1'), ('AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1', 'AFP_ENG_20030305.0918-EV34-1'), ('AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1', 'AFP_ENG_20030305.0918-EV35-1'), ('AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1', 'AFP_ENG_20030305.0918-EV12-1'), ('AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1', 'AFP_ENG_20030305.0918-EV34-1'), ('AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1', 'AFP_ENG_20030305.0918-EV35-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-1', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-1', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-11', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-11', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-14', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-14', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1'), ('AFP_ENG_20030305.0918-EV8-1', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1', 'AFP_ENG_20030305.0918-EV12-1'), ('AFP_ENG_20030305.0918-EV8-1', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1', 'AFP_ENG_20030305.0918-EV34-1'), ('AFP_ENG_20030305.0918-EV8-1', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1', 'AFP_ENG_20030305.0918-EV35-1'), ('AFP_ENG_20030305.0918-EV8-1', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1', 'AFP_ENG_20030305.0918-EV12-1'), ('AFP_ENG_20030305.0918-EV8-1', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1', 'AFP_ENG_20030305.0918-EV34-1'), ('AFP_ENG_20030305.0918-EV8-1', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1', 'AFP_ENG_20030305.0918-EV35-1'), ('AFP_ENG_20030305.0918-EV8-11', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1', 'AFP_ENG_20030305.0918-EV12-1'), ('AFP_ENG_20030305.0918-EV8-11', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1', 'AFP_ENG_20030305.0918-EV34-1'), ('AFP_ENG_20030305.0918-EV8-11', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1', 'AFP_ENG_20030305.0918-EV35-1'), ('AFP_ENG_20030305.0918-EV8-11', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1', 'AFP_ENG_20030305.0918-EV12-1'), ('AFP_ENG_20030305.0918-EV8-11', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1', 'AFP_ENG_20030305.0918-EV34-1'), ('AFP_ENG_20030305.0918-EV8-11', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1', 'AFP_ENG_20030305.0918-EV35-1'), ('AFP_ENG_20030305.0918-EV8-14', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1', 'AFP_ENG_20030305.0918-EV12-1'), ('AFP_ENG_20030305.0918-EV8-14', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1', 'AFP_ENG_20030305.0918-EV34-1'), ('AFP_ENG_20030305.0918-EV8-14', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1', 'AFP_ENG_20030305.0918-EV35-1'), ('AFP_ENG_20030305.0918-EV8-14', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1', 'AFP_ENG_20030305.0918-EV12-1'), ('AFP_ENG_20030305.0918-EV8-14', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1', 'AFP_ENG_20030305.0918-EV34-1'), ('AFP_ENG_20030305.0918-EV8-14', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1', 'AFP_ENG_20030305.0918-EV35-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-1', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1', 'AFP_ENG_20030305.0918-EV12-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-1', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1', 'AFP_ENG_20030305.0918-EV34-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-1', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1', 'AFP_ENG_20030305.0918-EV35-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-1', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1', 'AFP_ENG_20030305.0918-EV12-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-1', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1', 'AFP_ENG_20030305.0918-EV34-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-1', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1', 'AFP_ENG_20030305.0918-EV35-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-11', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1', 'AFP_ENG_20030305.0918-EV12-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-11', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1', 'AFP_ENG_20030305.0918-EV34-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-11', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1', 'AFP_ENG_20030305.0918-EV35-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-11', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1', 'AFP_ENG_20030305.0918-EV12-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-11', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1', 'AFP_ENG_20030305.0918-EV34-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-11', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1', 'AFP_ENG_20030305.0918-EV35-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-14', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1', 'AFP_ENG_20030305.0918-EV12-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-14', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1', 'AFP_ENG_20030305.0918-EV34-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-14', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV28-1', 'AFP_ENG_20030305.0918-EV35-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-14', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1', 'AFP_ENG_20030305.0918-EV12-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-14', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1', 'AFP_ENG_20030305.0918-EV34-1'), ('AFP_ENG_20030305.0918-EV21-1', 'AFP_ENG_20030305.0918-EV8-14', 'AFP_ENG_20030305.0918-EV20-1', 'AFP_ENG_20030305.0918-EV29-1', 'AFP_ENG_20030305.0918-EV35-1')]
