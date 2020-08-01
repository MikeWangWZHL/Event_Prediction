# from tqdm import tqdm
# x = 0
# for i in tqdm(range(10000)):
#     x += 1
import nltk
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# nltk.download('words')
# nltk.download('maxent_ne_chunker')
sentence = "I love my life, but I hate stupid people in Urbana, and UIUC"
tokens = nltk.word_tokenize(sentence)
tagged = nltk.pos_tag(tokens)
entities = nltk.chunk.ne_chunk(tagged)
print(tokens)
print(tagged)
print(entities)
