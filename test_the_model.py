from keras.models import load_model
import tensorflow as tf
import numpy as np
import pickle
import random

# imoport the spaCy nlp model
import spacy
nlp = spacy.load('en_core_web_sm')


# load saved ml model, list of tokens, and dictionary of outputs
model = load_model('chatbot_model.h5')

with open(f'pickles/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)
with open(f'pickles/intent_doc.pkl', 'rb') as file:
    intent_doc = pickle.load(file)
with open(f'pickles/trg_index_word.pkl', 'rb') as file:
    trg_index_word = pickle.load(file)


def response(sentence):
    sent_seq = []
    doc = nlp(repr(sentence))
    
    # split the input sentences into words
    for token in doc:
        if token.text in tokenizer.word_index:
            sent_seq.append(tokenizer.word_index[token.text])

        # handle unknown words
        else:
            sent_seq.append(tokenizer.word_index['<unk>'])

    sent_seq = tf.expand_dims(sent_seq, 0)
    
    # predict the intent of input sentences
    pred = model(sent_seq)
    pred_class = np.argmax(pred.numpy(), axis=1)
    
    # choose random response from provided options
    return random.choice(intent_doc[trg_index_word[pred_class[0]]]), trg_index_word[pred_class[0]]


# chat with bot
print("Enter 'quit' to break the loop.")
while True:
    inp = input('You: ')
    if inp.lower() == 'quit':
        break
    res, typ = response(inp)
    print('Bot: {} -- TYPE: {}'.format(res, typ))
    print()
