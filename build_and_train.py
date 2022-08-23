from keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional, GlobalMaxPooling1D
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from spacy.lang.en.stop_words import STOP_WORDS
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings(action = 'ignore')

def build_BiLSTM(size_of_vocabulary):
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
    print('Test Loss: {}'.format(loss))
    return acc, loss

# sim = word2vec('SentiNet/models/data.txt',"Skip Gram", "neural", "net")
# print(sim)





