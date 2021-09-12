#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Kagle inclass https://www.kaggle.com/c/simplesentiment/overview

# In[1]:


import os
from pathlib import Path

import pandas as pd
import numpy as np

from itertools import product
#import warnings
#from tqdm import tqdm

import pickle as pkl
import re


# In[2]:


#import nltk
import nltk.stem as st


# In[3]:


#from gensim.sklearn_api import W2VTransformer
#import gensim.downloader as api


# In[4]:


#from sklearn.pipeline import Pipeline
#from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression#, SGDClassifier
#from sklearn.svm import LinearSVC
#from sklearn.svm import SVC
#from sklearn.ensemble import RandomForestClassifier

#from sklearn.model_selection import GridSearchCV, cross_val_score 

#from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


# In[ ]:





# In[5]:


PATH_DATA = os.path.join(Path.cwd(), 'data')
PATH_SUBM = os.path.join(Path.cwd(), 'submissions')
PATH_MODL = os.path.join(Path.cwd(), 'models')


# In[ ]:





# In[21]:


df = pd.read_csv(os.path.join(PATH_DATA, 'products_sentiment_train.tsv'), 
                    header = None, 
                    index_col = None,
                    sep = '\t',
                   )
df.columns = ['text', 'target']
df.shape


# In[7]:


clean_text = lambda x: re.sub(r"\s+", ' ', 
                              re.sub(r"[\d+]", '',
                                     re.sub(r"[^\w\s]", '', x.lower()).strip()
                                    )
                             )

lemm = st.WordNetLemmatizer()
lem_text = lambda x: ' '.join([lemm.lemmatize(el) for el in x.split()])

#stemm = st.ISRIStemmer()
stemm = st.RSLPStemmer()
stem_text = lambda x: ' '.join([stemm.stem(el) for el in x.split()])


# In[8]:


df['text_cl'] = df.text.map(clean_text)
df['text_cl'] = df.text_cl.map(lem_text)
df['text_cl'] = df.text_cl.map(stem_text)


# In[ ]:





# ### Создаем и сохраняем модель и токенайзер для загрузки в демонстрацию на flask

# In[9]:


vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), 
                             max_df=0.75, min_df=1, 
                             stop_words=None
                            )
vectorizer.fit(df.text_cl)
train = vectorizer.transform(df.text_cl)


# In[10]:


model = LogisticRegression(penalty = 'l2',
                           solver = 'liblinear',
                           C = 1.7189473684210526,
                           class_weight = {0: 0.5894081632653061, 1: 0.41059183673469385},
                           max_iter = 75,
                           random_state = 111111, 
)
model.fit(train, df.target)


# In[11]:


with open(os.path.join(PATH_MODL, 'tfidf_lr_model.pkl'), 'wb') as fd:
    pkl.dump(model, fd)
    
with open(os.path.join(PATH_MODL, 'tfidf_lr_token.pkl'), 'wb') as fd:
    pkl.dump(vectorizer, fd)


# In[ ]:





# In[13]:


pred_train_tfidf = model.predict(train)


# In[14]:


roc_auc_score(df.target, pred_train_tfidf)


# In[15]:


confusion_matrix(df.target, pred_train_tfidf)


# In[ ]:




