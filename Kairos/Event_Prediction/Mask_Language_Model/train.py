# from transformers import pipeline
# from pprint import pprint
# nlp = pipeline("fill-mask")
# pprint(nlp(f"HuggingFace is creating a {nlp.tokenizer.mask_token} that the community uses to solve NLP tasks."))

# from transformers import AutoTokenizer, AutoModelForMaskedLM
# import torch

# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
# model = AutoModelForMaskedLM.from_pretrained("distilbert-base-cased")

# print(model)

# # sequence = f"Distilled models are smaller than the models they mimic. Using them instead of the large versions would help {tokenizer.mask_token} our carbon footprint."


# input_ = tokenizer.encode(sequence, return_tensors="pt")



from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

sequence = f"<Michael Crichton> divorced in <Place> on <2002-09><SEP> <Anne-Marie> filed suit against <Michael Crichton> before <Adjudicator> court or judge for <Crime> crime in <Place> place on <2002-09><SEP>  {tokenizer.mask_token} gave <31 million dollars> money to <Anne-Marie> for the benefit of <Beneficiary> at <Place> place on <NaN>"



input_ids = tokenizer(sequence, return_tensors="pt")["input_ids"]
# print(input_ids)

mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1] #where is the position in the input
# print(mask_token_index)

outputs = model(input_ids, labels=input_ids) # (loss,prediction_scores)

# print(outputs)
mask_token_outputs = outputs[1][0][mask_token_index] # find the output probabilities for the [MASK] postition
# print(mask_token_outputs)
top_5_tokens = torch.topk(mask_token_outputs, 5, dim=1).indices[0].tolist()

print(top_5_tokens)

for token in top_5_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))
