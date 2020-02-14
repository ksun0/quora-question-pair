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

df_train = pd.read_csv('train.csv')#, sep='\|\|', header=None, skiprows=1, names=["ID","Text"])
#df_train.head()

#df_train.describe(include ='all')
#df_train['qid1'].value_counts().plot(kind = 'bar', rot =0)

def evaluate_features(X, y, clf=None):
    if clf is None:
        clf = LogisticRegression()
    
    probas = cross_val_predict(clf, X, y, cv=StratifiedKFold(random_state=8), 
                              n_jobs=-1, method='predict_proba', verbose=2)
    pred_indices = np.argmax(probas, axis=1)
    classes = np.unique(y)
    preds = classes[pred_indices]
    print('Log loss: {}'.format(log_loss(y, probas)))
    print('Accuracy: {}'.format(accuracy_score(y, preds)))
    #skplt.plot_confusion_matrix(y, preds)
    
count_vectorizer = CountVectorizer(analyzer="word", tokenizer=nltk.word_tokenize,preprocessor=None, stop_words='english', max_features=None)  

df_q1q2 = df_train.dropna(how = 'any', axis =0)
#print(df_q1q2['question1'].isnull().values.any())
#print(df_q1q2.head())



def LinearizeWords(column):
    words = []
    for rows in column:
        for sentences in nltk.sent_tokenize(rows):
            words += nltk.word_tokenize(sentences)
            return words

X1 = LinearizeWords(df_q1q2['question1'])
X2 = LinearizeWords(df_q1q2['question2'])

def Word2VecModel(sentences, location):
    if os.path.exists(location):
        print('Found {}'.format(location))
        model = gensim.models.Word2Vec.load(location)
        return model
    
    print('{} not found. training model'.format(location))
    model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
    print('Model done training. Saving to disk')
    model.save(location)
    return model

w2vec1 = Word2VecModel(X1,'w2vModel1')
w2vec2 = Word2VecModel(X2,'w2vModel2')

def WordTokenizer(columns):
    #Xt = np.zeros((columns.shape[0],1))
    Xt = []
    for i,rows in enumerate(columns):
        Xtt = []
        for sent in nltk.sent_tokenize(rows):
            Xtt += nltk.word_tokenize(sent)
        Xt.append(np.array(Xtt))
    return np.array(Xt)

X1t = WordTokenizer(df_q1q2['question1'])
print('tokenized1 shape', X1t.shape)

X2t = WordTokenizer(df_q1q2['question2'])
print('tokenized2 shape', X2t.shape)

def MeanEmbedVector(X, Word2VecM):
    dim = len(Word2VecM.wv.syn0[0])
    z = np.zeros((X.shape[0],1))


    for i,words in enumerate(X):
        for w in words:
            if w in Word2VecM.wv:
                z[i,0] = np.mean(Word2VecM.wv[w], axis = 0)
        #print(np.mean([Word2VecM.wv[w] for w in words if w in Word2VecM.wv], axis = 0))

    return z

meanEmbed1 = MeanEmbedVector(X1t, w2vec1)
meanEmbed2 = MeanEmbedVector(X2t, w2vec2)

features = np.concatenate((meanEmbed1,meanEmbed2), axis = 1)
evaluate_features(features, df_q1q2['is_duplicate'].values.ravel())
#print(meanEmbed1.shape)

