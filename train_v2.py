from transformers import BertTokenizer, BertConfig
from transformers import AdamW
from modeling_bert import BertForSequenceClassification
# from transformers import BertForSequenceClassification

import torch 
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine
from torch import nn

# from gen_event_entity_dict import gen_event_entity_role_dict
from prepare_input_v2 import prepare_input

"""set up type dicts"""
# raw_file = 'all_files_update.json'
# event_type_dict,entity_type_dict,role_type_dict = gen_event_entity_role_dict(raw_file)
# class_size = len(event_type_dict)

event_type_dict = {'Divorce': 0, 'EndPosition': 1, 'Acquit': 2, 'Meet': 3, 'Die': 4, 'Extradite': 5, 'Sue': 6, 'Elect': 7, 'Convict': 8, 'TransferOwnership': 9, 'Marry': 10, 'Attack': 11, 'StartPosition': 12, 'ArrestJail': 13, 'ReleaseParole': 14, 'Nominate': 15, 'Transport': 16, 'Fine': 17, 'Sentence': 18, 'TrialHearing': 19, 'BeBorn': 20, 'Pardon': 21, 'Demonstrate': 22, 'Execute': 23, 'StartOrg': 24, 'PhoneWrite': 25, 'Appeal': 26, 'Injure': 27, 'ChargeIndict': 28, 'TransferMoney': 29, 'Null':30}
entity_type_dict = {'ORG': 0, 'LOC': 1, 'VEH': 2, 'WEA': 3, 'GPE': 4, 'FAC': 5, 'PER': 6, 'CLS':7, 'SEP':8, 'PAD':9, 'UNK':10}
for e in entity_type_dict.keys():
    entity_type_dict[e] += 1
role_type_dict = {'Target': 0, 'Plaintiff': 1, 'Person': 2, 'Seller': 3, 'Time': 4, 'Recipient': 5, 'Instrument': 6, 'Artifact': 7, 'Adjudicator': 8, 'Prosecutor': 9, 'Agent': 10, 'Beneficiary': 11, 'Attacker': 12, 'Victim': 13, 'Money': 14, 'Buyer': 15, 'Docid': 16, 'Crime': 17, 'Giver': 18, 'Sentence': 19, 'Org': 20, 'Defendant': 21, 'Position': 22, 'Vehicle': 23, 'Destination': 24, 'Origin': 25, 'Place': 26, 'Entity': 27, 'Price':28, 'CLS':29, 'SEP':30, 'PAD':31} 
for r in role_type_dict.keys():
    role_type_dict[r] += 1
entity_type_dict['OTHER'] = 0
role_type_dict['Other'] = 0
# print(entity_type_dict)
# print(role_type_dict)
idx_to_event = {value:key for key,value in event_type_dict.items()}
idx_to_role = {value:key for key,value in role_type_dict.items()}
idx_to_entity = {value:key for key,value in entity_type_dict.items()}


print(event_type_dict)
# print(entity_type_dict)
# print(role_type_dict)
"""set up tokenizer"""
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

"""import data"""

input_ids,attention_masks,role_type_ids,entity_type_ids,labels = prepare_input('test_temp_three_level.json',event_type_dict,entity_type_dict,role_type_dict,tokenizer)


"""split train and val dataset"""
from torch.utils.data import TensorDataset, random_split

# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks,role_type_ids,entity_type_ids, labels)

# Create a 90-10 train-validation split.

# Calculate the number of samples to include in each set.
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))


"""prepare dataloader"""
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# The DataLoader needs to know our batch size for training, so we specify it 
# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# size of 16 or 32.
batch_size = 16

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )


"""setup model, optimizer"""
# configuration = BertConfig.from_pretrained('bert-base-cased')
# print(configuration)
# quit()
pretrain_config = BertConfig.get_config_dict('bert-base-cased')[0]
pretrain_config['entity_type_size'] = len(entity_type_dict)
pretrain_config['role_type_size'] = len(role_type_dict)
pretrain_config['class_size'] = len(event_type_dict)
pretrain_config['chunk_size_feed_forward'] = 0
pretrain_config['add_cross_attention'] = False
pretrain_config['use_return_dict'] = True
pretrain_config['output_hidden_states'] = True
pretrain_config['num_labels'] = len(event_type_dict)
print(len(event_type_dict))


configuration = BertConfig.from_dict(pretrain_config)
configuration.update(pretrain_config)

# print(configuration)
# quit()
model = BertForSequenceClassification.from_pretrained('bert-base-cased',config=configuration)
# print(model)

optimizer = AdamW(model.parameters(), lr=1e-5)
# model.config = model.config.from_dict(pretrain_config)


"""setup epoch, scheduler"""
from transformers import get_linear_schedule_with_warmup

# Number of training epochs. The BERT authors recommend between 2 and 4. 
# We chose to run for 4, but we'll see later that this may be over-fitting the
# training data.
epochs = 8

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs
print("size of train_dataloader:" ,len(train_dataloader))
# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
"""helper functions"""
import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


"""training step"""

import random
import numpy as np

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    # print(pred_flat)
    labels_flat = labels.flatten()
    # print(labels_flat)
    type_pred = [idx_to_event[i] for i in pred_flat]
    type_groundtruth = [idx_to_event[i] for i in labels_flat]
    # print('predict:',type_pred,'ground truth:',type_groundtruth)
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# device = 'cuda:2'
device = 'cpu'

# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

# For each epoch...
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        
        # Progress update every 40 batches.
        if step % 2 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        # quit()
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_role_type_ids = batch[2].to(device)
        b_entity_type_ids = batch[3].to(device)
        b_labels = batch[4].to(device)
        # print(b_input_ids)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        model.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch).
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # It returns different numbers of parameters depending on what arguments
        # arge given and what flags are set. For our useage here, it returns
        # the loss (because we provided labels) and the "logits"--the model
        # outputs prior to activation.
        # print(model)
        
        outputs = model(b_input_ids, attention_mask=b_input_mask,role_type_ids=b_role_type_ids,entity_type_ids=b_entity_type_ids, labels=b_labels)
        loss = outputs[0] 


        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.

        total_train_loss += loss.item()


        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using 
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_role_type_ids = batch[2].to(device)
        b_entity_type_ids = batch[3].to(device)
        b_labels = batch[4].to(device)
        
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            outputs = model(b_input_ids, attention_mask=b_input_mask,role_type_ids=b_role_type_ids,entity_type_ids=b_entity_type_ids, labels=b_labels)
            # outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0] 
            logits = outputs[1]

            
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.

        total_eval_accuracy += flat_accuracy(logits, label_ids)
        

    # Report the final accuracy for this validation run.

    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))




"""save model"""
import os

# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

output_dir = './model_save/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)



# print(pretrain_config)
# print('')
# print('config after:\n',model.config)

# my_embedding_layers = BertEmbeddingsAdd(model.config)
# model.bert.embeddings = my_embedding_layers


# print('model architecture:',model)
# print('model embedding layer:',model.bert.embeddings)
# print('token_embedding layer:',model.bert.embeddings.word_embeddings)
# print('input embeddings:',model.get_input_embeddings())

# inputs = tokenizer(f"<jessica lynch> was injured by <Agent> using <Instrument> at <Place> place on <NaN>", return_tensors="pt")

# print('input string:',inputs)

# labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1

# # print('output_hidden_states:',model.config.output_hidden_states)

# #TODO: get these embedding ids
# role_type_ids = torch.tensor([0,1,1,1,1,1,1,0])
# entity_type_ids = torch.tensor([0,1,1,1,1,1,1,0])

# with torch.no_grad():
#     # inputs_embeds = my_embedding_layer(input_ids = inputs['input_ids'],role_type_ids=role_type_ids,entity_type_ids = entity_type_ids)
#     # print(inputs_embeds)
#     outputs = model(**inputs, labels=labels)
#     # outputs = model(labels=labels,inputs_embeds = inputs_embeds)
#     loss, logits, hidden_states = outputs[:3]


# print(loss)
# print('output probs:',logits)
# print('layer number:',len(hidden_states))
# print('batch number:',len(hidden_states[0]))
# print('token number:',len(hidden_states[0][0]))
# print('hiddent unit number:',len(hidden_states[0][0][0]))
# print('initial embedding:',hidden_states[0][0])

"""visualize embedding"""
# # For the 5th token in our sentence, select its feature values from layer 5.
# token_i = 2
# layer_i = 0
# vec = hidden_states[layer_i][0][token_i]

# # Plot the values as a histogram to show their distribution.
# plt.figure(figsize=(10,10))
# plt.hist(vec, bins=200)
# plt.show()