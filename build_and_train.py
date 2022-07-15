from keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional, GlobalMaxPooling1D
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import adam_v2
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import librosa
from spacy.lang.en.stop_words import STOP_WORDS

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings(action = 'ignore')
import gensim
from gensim.models import Word2Vec
 
def word2vec(file_path, model_name, word1, word2):
    #  Reads text file
    sample = open(file_path)
    s = sample.read()
    # Replaces escape character with space
    f = s.replace("\n", " ")
    data = []
    # iterate through each sentence in the file
    for i in sent_tokenize(f):
        temp = [] 
        # tokenize the sentence into words
        for j in word_tokenize(i):
            temp.append(j.lower())
        data.append(temp)
    try:
        model_name == "CBOW" or model_name == "Skip Gram"
        if model_name == "CBOW":
            model = Word2Vec(data, min_count = 1, vector_size = 100, window = 5)
            sim = model.wv.similarity(word1, word2)
            output = "Cosine similarity between {} " + "and {} - CBOW : ".format(word1, word2) + str(sim)
            return output
        elif model_name =="Skip Gram":
            model = Word2Vec(data, min_count = 1, vector_size = 100, window = 5, sg = 1)
            sim = model.wv.similarity(word1, word2)
            output = "Cosine similarity between {} " + "and {} - CBOW : ".format(word1, word2) + str(sim)
            return output
    except:
        return ValueError("Model used not available. Available models are 'CBOW' or 'Skip Gram'.")

def build_BiLSTM(input_dim, size_of_vocabulary):
    model = Sequential()
    #embedding layer
    model.add(Embedding(size_of_vocabulary,128,input_length=120))
    #lstm layer
    model.add(Bidirectional(LSTM(64,return_sequences=True,dropout=0.2)))
    #Global Maxpooling
    model.add(GlobalMaxPooling1D())
    #Dense Layer
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(1,activation='sigmoid'))
    #Add loss function, metrics, optimizer
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    #Adding callbacks
    es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=3)
    mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True,verbose=1)
    #summary
    model.summary()
    return model, es, mc

def fit(model, X_train, X_valid, y_train, y_valid, es, mc):
    history = model.fit(X_train,y_train,batch_size=128,epochs=4, 
    validation_data=(X_valid,y_valid),verbose=1,callbacks=[es,mc])
    return model, history

def plot_training(h):
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(h.history["loss"], label="train_loss")
    # plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(h.history["accuracy"], label="train_acc")
    # plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.show()

def get_accuracy_and_loss(model, X_test, y_test):
    loss,acc = model.evaluate(X_test,y_test)
    print('Test Accuracy: {}%'.format(acc*100)) 
    print('Test Loss: {}%'.format(loss))
    return acc, loss

# sim = word2vec('SentiNet/models/data.txt',"Skip Gram", "neural", "net")
# print(sim)





