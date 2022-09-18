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


import nltk
import nltk.stem as st


# In[3]:


nltk.download('omw-1.4')


# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression#, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


# In[ ]:





# # Выставляем переменные

# In[5]:


PATH_DATA = os.path.join(Path.cwd(), 'data')
PATH_SUBM = os.path.join(Path.cwd(), 'submissions')
PATH_MODL = os.path.join(Path.cwd(), 'models')


# In[ ]:





# # Загрузка и очистка данных

# In[6]:


#df = pd.read_csv(os.path.join(PATH_DATA, 'products_sentiment_train.tsv'), 
df = pd.read_csv(os.path.join(PATH_DATA, 'ru_train.csv'), 
                    #header = None, 
                    #index_col = None,
                    #sep = '\t',
                   )
#df.columns = ['text', 'target']
df.shape


# In[7]:


df.head()


# In[8]:


df['target'] = df.rating.apply(lambda x: 1 if int(x) >= 4 else 2)


# In[9]:


clean_text = lambda x: re.sub(r"\s+", ' ', 
                              re.sub(r"[\d+]", '',
                                     re.sub(r"[^\w\s]", '', x.lower()).strip()
                                    )
                             )

# приведение к начальным формам
lemm = st.WordNetLemmatizer()
lem_text = lambda x: ' '.join([lemm.lemmatize(el) for el in x.split()])

#stemm = st.ISRIStemmer()
stemm = st.RSLPStemmer()
stem_text = lambda x: ' '.join([stemm.stem(el) for el in x.split()])


# In[10]:


get_ipython().run_cell_magic('time', '', "df['text_cl'] = df.review.map(clean_text)\ndf['text_cl'] = df.text_cl.map(lem_text)\ndf['text_cl'] = df.text_cl.map(stem_text)")


# Перемешиваем отзывы

# In[11]:


df = df.sample(frac=1).reset_index(drop=True)
df.drop(['Unnamed: 0'], axis = 1, inplace = True)
df.head(3)


# In[ ]:





# # Создание и сохранение модели и токенайзер для загрузки в демонстрацию на flask

# Очищенные отзывы векторизуем через tf-idf.   
# Посимвольно, длинною 3 или 4 без исключения стопслов.    
# мин частота - 1, макс частота - 0.75.   
# Полученные векторы в LogReg с подобранными параметрами

# In[12]:


vectorizer = TfidfVectorizer(analyzer = 'char_wb', ngram_range = (3, 4), 
                             max_df = 0.75, min_df = 1, 
                             stop_words = None
                            )
vectorizer.fit(df.text_cl)
train = vectorizer.transform(df.text_cl)


# In[13]:


model = LogisticRegression(penalty = 'l2',
                           solver = 'liblinear',
                           C = 1.7189473684210526,
                           class_weight = {1: 0.5894081632653061, 2: 0.41059183673469385},
                           max_iter = 75,
                           random_state = 111111, 
)
model.fit(train, df.target)


# In[14]:


with open(os.path.join(PATH_MODL, 'tfidf_lr_model.pkl'), 'wb') as fd:
    pkl.dump(model, fd)
    
with open(os.path.join(PATH_MODL, 'tfidf_lr_token.pkl'), 'wb') as fd:
    pkl.dump(vectorizer, fd)


# In[ ]:





# # Посмотрим на результат обучения на трейне

# In[15]:


pred_train_tfidf = model.predict(train)


# In[16]:


roc_auc_score(df.target, pred_train_tfidf)


# In[17]:


confusion_matrix(df.target, pred_train_tfidf)


# In[ ]:




