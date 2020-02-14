import os
import numpy as  np

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

import gensim

import scikitplot.plotters as skplt

import nltk

#from xgboost import XGBClassifier1996


import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam

df_train = pd.read_csv('train.csv')

numOfWords = 2000
tokenizer = Tokenizer(num_words = numOfWords)
#print(type(df_train[['question1', 'question2']]))
#df1 = df_train['question1'].values#, 'question2']]
#df2 = df_train['question2'].values


df = (df_train["question1"].map(str) + df_train["question2"]).astype(str)

#print(df.head())
#features = np.concatenate((df1,df2))

tokenizer.fit_on_texts(df.values)

X = tokenizer.texts_to_sequences(df.values)
X = pad_sequences(X, maxlen = 2000)

embed_dim = 128
lstm_out = 196

ckpt_callback = ModelCheckpoint('keras_model',monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

model = Sequential()
model.add(Embedding(numOfWords, embed_dim, input_length = X.shape[1]))
model.add(LSTM(lstm_out, recurrent_dropout = 0.2, dropout = 0.2))
model.add(Dense(2, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['categorical_crossentropy'])
print(model.summary())

Y = pd.get_dummies(df_train['is_duplicate']).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify=Y)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

batch_size = 32
model.fit(X_train, Y_train, epochs=8, batch_size=batch_size, validation_split=0.2, callbacks=[ckpt_callback])

model = load_model('keras_model')
probas = model.predict(X_test)



pred_indices = np.argmax(probas, axis=1)
classes = np.array(range(1, 10))
preds = classes[pred_indices]
print('Log loss: {}'.format(log_loss(classes[np.argmax(Y_test, axis=1)], probas)))
print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(Y_test, axis=1)], preds)))
