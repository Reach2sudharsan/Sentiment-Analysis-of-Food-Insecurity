import pandas as pd
import matplotlib.pyplot as plt
import re
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from wordcloud import WordCloud
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

def retrieve_data():
    train = pd.read_csv('training_data/Train.csv')
    valid = pd.read_csv('training_data/Valid.csv')
    test = pd.read_csv('training_data/Test.csv')
    train.info()
    valid.info()
    test.info()
    return train, valid, test

def Preprocessing(text):
    text = re.sub(r'[^\w\s]','',text)
    text = text.lower()
    text = [w for w in text.split(' ') if w not in stopwords.words('english')]
    text = [WordNetLemmatizer().lemmatize(token) for token in text]
    text = [WordNetLemmatizer().lemmatize(token,pos='v') for token in text]
    text = " ".join(text)
    return text

def preprocess_train_valid_test(train, valid, test):
    train['text'] = train.text.apply(lambda x:Preprocessing(x))
    valid['text'] = valid.text.apply(lambda x:Preprocessing(x))
    test['text']= test.text.apply(lambda x:Preprocessing(x)) 
    return train, valid, test #train['label'].value_counts()

def split_train_valid_test(train, valid, test):
    X_train = train['text']
    X_valid = valid['text']
    X_test = test['text']
    y_train = train['label']
    y_valid = valid['label']
    y_test = test['label']
    return (X_train, X_valid, X_test), (y_train, y_valid, y_test)

def view_text_length(X_train, X_valid, X_test):
    plt.figure(figsize=(16,20))
    plt.style.use('fivethirtyeight')

    plt.subplot(3,1,1)
    train_len = [len(l) for l in X_train]
    plt.hist(train_len,bins=50)
    plt.title('Distribution of train text length')
    plt.xlabel('Length')

    plt.subplot(3,1,2)
    valid_len = [len(l) for l in X_valid]
    plt.hist(valid_len,bins=50,color='green')
    plt.title('Distribution of valid text length')
    plt.xlabel('Length')

    plt.subplot(3,1,3)
    test_len = [len(l) for l in X_test]
    plt.hist(test_len,bins=50,color='red')
    plt.title('Distribution of test text length')
    plt.xlabel('Length')

    plt.show()

def view_wordclouds(train):
    plt.figure(figsize=(20,20))
    pos_freq = FreqDist(' '.join(train[train['label'] == 1].text).split(' '))
    wc = WordCloud().generate_from_frequencies(frequencies=pos_freq)
    plt.imshow(wc,interpolation='bilinear')
    plt.title('Positive Review Common Text')
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(20,20))
    neg_freq = FreqDist(' '.join(train[train['label'] == 0].text).split(' '))
    wc = WordCloud().generate_from_frequencies(frequencies=neg_freq)
    plt.imshow(wc,interpolation='bilinear')
    plt.title('Negative Review Common Text')
    plt.axis('off')
    plt.show()

def view_common_words(train):
    pos_freq = FreqDist(' '.join(train[train['label'] == 1].text).split(' '))
    plt.figure(figsize=(20,6))
    pos_freq.plot(50,cumulative=False,title='Positive Review Common Text')
    plt.show()

    neg_freq = FreqDist(' '.join(train[train['label'] == 0].text).split(' '))
    plt.figure(figsize=(20,6))
    neg_freq.plot(50,cumulative=False,title='Negative Review Common Text',color='red')
    plt.show()

def prepare_data(X_train, X_valid, X_test):
    #Tokenize the sentences
    tokenizer = Tokenizer()
    #preparing vocabulary
    tokenizer.fit_on_texts(X_train)
    #converting text into integer sequences
    X_train = tokenizer.texts_to_sequences(X_train)
    X_valid = tokenizer.texts_to_sequences(X_valid)
    X_test = tokenizer.texts_to_sequences(X_test)
    #padding to prepare sequences of same length
    X_train=pad_sequences(X_train,maxlen=120)
    X_valid=pad_sequences(X_valid,maxlen=120)
    X_test=pad_sequences(X_test,maxlen=120)

    size_of_vocabulary = len(tokenizer.word_index)+1
    print("Vocabulary Size: " + str(size_of_vocabulary))

    return X_train, X_valid, X_test, size_of_vocabulary



