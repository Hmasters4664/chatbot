import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import pandas as pd
import pickle
import random
import json

intents_file = open('intent.json').read()
intents = json.loads(intents_file)

words = []
classes = []
documents = []
ignore_words = ['?']

#Loop through intent json file and tokenize intent using ntlk

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        # add words to the list
        words.extend(w)
        # add document to the corpus
        documents.append((w,intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# sort classes
classes = sorted(list(set(classes)))

# training data
training = []
# create an empty array for output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

for doc in documents:
    # initialize the B.O.W
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]

    #stem each word - create base word, in attempt to represent related words
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

    # create our bag of words array with 1, if word mach found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # 
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag,output_row])

#shuffle
random.shuffle(training)
training = np.array(training)

# create train and test lists. X - patterns, Y- intents
train_x = list(training[:,0])
train_y = list(training[:,1])

# Create model - 3 layers. First layer 128 neurons, seconf layer 64 neurons and 3rd output layer contians number of nuerons equal to intents
model = Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])

model.fit(np.array(train_x),np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')
print("model is created")
