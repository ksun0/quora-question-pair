
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

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
df_train.head()

#df_train.describe(include ='all')
#df_train['qid1'].value_counts().plot(kind = 'bar', rot =0)

def evaluate_features(X, y, clf=None): #function to display and predict the error rate
    if clf is None:
        clf = LogisticRegression()
    
    probas = cross_val_predict(clf, X, y, cv=StratifiedKFold(random_state=8), 
                              n_jobs=-1, method='predict_proba', verbose=2)

    #pred_indices = np.argmax(probas, axis=1)
    #classes = np.unique(y)
    #preds = classes[pred_indices]
    print('Log loss: {}'.format(log_loss(y, probas)))
    print('Accuracy: {}'.format(accuracy_score(y, preds)))
    #skplt.plot_confusion_matrix(y, preds)
    
count_vectorizer = CountVectorizer(analyzer="word", tokenizer=nltk.word_tokenize,preprocessor=None, stop_words='english', max_features=None)  

df_q1q2 = df_train.dropna(how = 'any', axis =0)
print(df_q1q2['question1'].isnull().values.any())
print(df_q1q2.head())

bag_of_words1 = count_vectorizer.fit_transform(df_q1q2['question1'])
bag_of_words2 = count_vectorizer.fit_transform(df_q1q2['question2'])
svd = TruncatedSVD(n_components=25, n_iter=25, random_state=12)
truncated_bag_of_words1 = svd.fit_transform(bag_of_words1)
truncated_bag_of_words2 = svd.fit_transform(bag_of_words2)
print(truncated_bag_of_words1.shape)
features = np.concatenate((truncated_bag_of_words1,truncated_bag_of_words2), axis = 1)
print(features.shape)
evaluate_features(features, df_q1q2['is_duplicate'].values.ravel())
#evaluate_features(truncated_bag_of_words, df_train['Class'].values.ravel(),RandomForestClassifier(n_estimators=1000, max_depth=5, verbose=1))