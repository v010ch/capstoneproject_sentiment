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
import warnings
from tqdm import tqdm

import re


# In[2]:


import nltk
#nltk.download('movie_reviews')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('rslp')

import nltk.stem as st


# In[3]:


import spacy

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.sklearn_api import W2VTransformer


# In[76]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score 

#from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


# In[45]:


import xgboost as xgb
from lightgbm import LGBMClassifier


# In[ ]:





# In[6]:


PATH_DATA = os.path.join(Path.cwd(), 'data')
PATH_SUBM = os.path.join(Path.cwd(), 'submissions')


# In[ ]:





# In[ ]:





# In[8]:


df = pd.read_csv(os.path.join(PATH_DATA, 'products_sentiment_train.tsv'), 
                    header = None, 
                    index_col = None,
                    sep = '\t',
                   )
df.columns = ['text', 'target']
df.shape


# In[9]:


df.head()


# In[10]:


df.target.value_counts()


# Классы явно не сбалансированы.   
# Т.к. мы работаем с текстами есть следующие варианты:   
#     1. добавить негативные примеры. + увеличение выборки, баланс классов. - необходимо найти размеченную, подходящую под тематику, выборку   
#     2. дублировать некоторые строки, например те, в которых ошибается модель   
#     3. настроить параметр class_weights
#     

# ### Приводим к строчным (на всякий) и очищаем по порядку:

# In[11]:


# - все спецсимволы
# - все цифры
# - заменяем множественные пробелы на единичные


# In[12]:


clean_text = lambda x: re.sub(r"\s+", ' ', 
                              re.sub(r"[\d+]", '',
                                     re.sub(r"[^\w\s]", '', x.lower()).strip()
                                    )
                             )


# In[13]:


df['text_cl'] = df.text.map(clean_text)


# In[ ]:





# In[14]:


#tagged_data[:5]


# In[15]:


#lem_text("don't"), lem_text("i'll")


# In[16]:


#stem_text("don't"), stem_text("i'll")


# ### Лемматизация & стемминг

# In[17]:


lemm = st.WordNetLemmatizer()
lem_text = lambda x: ' '.join([lemm.lemmatize(el) for el in x.split()])


stemm = st.RSLPStemmer()
stem_text = lambda x: ' '.join([stemm.stem(el) for el in x.split()])


# In[18]:


df['text_cl'] = df.text.map(clean_text)
df['text_cl'] = df.text_cl.map(lem_text)
df['text_cl'] = df.text_cl.map(stem_text)


# In[19]:


df['text_cl'] = df.text_cl.map(lambda x: x.split())


# In[20]:


df.head()


# In[ ]:





# In[ ]:





# ### Подготавливаем к обучению по сетке

# In[21]:


train, test, train_target, test_target = train_test_split(df.text_cl, df.target, 
                                                          test_size = 0.2, 
                                                          stratify = df.target, 
                                                          random_state = 52138,
                                                         )


# In[ ]:





# ### Параметры для перебора

# In[24]:


pipe_cnt = Pipeline([#('w2v', W2VTransformer(size=100, min_count=1, seed=1)),
                     ('clf', LogisticRegression()),
                      ])


# In[25]:


parameters_v0 = {
        'clf': [LogisticRegression(), 
                LinearSVC(),
                #SGDClassifier(), 
                #RandomForestClassifier(),
                #xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0),
                #xgb.XGBClassifier(eval_metric = 'auc'),
               ],
    }


# In[60]:


parameters = [
            {'clf': [LogisticRegression()]}, 
            {'clf': [LinearSVC(max_iter = 1500)]},
            #{'clf': [SGDClassifier()]}, 
            #{'clf': [RandomForestClassifier()]},
            #{'clf': [xgb.XGBClassifier(eval_metric = 'auc', use_label_encoder=False)]},
                     #xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0),
            #{'clf': [LGBMClassifier()]},
]


# In[63]:


par_size = [300, 200,]# 100, 50]#10, 20, 30, 50, ]
par_window = [5,]# 4, 3,]# 2]
#par_hs = [1,]# 0]
par_niter = [50, 40,]# 30,]# 75,]# 100]#3, 5, 10, 50]
par_min_count = [5, 3,]# 10, 2, 1]
len(list(product(par_size, 
                 par_window, 
                 #par_hs, 
                 par_niter, 
                 par_min_count)))


# In[97]:


# Трансфорование данных (отзыва) в вектор
# требуется исключить слова, которые не встречались, т.к. выдаст ошибку
# и полученные вектораотзыва усредняем по столбцам
def transform_x2v(inp_model, inp_series):
    transformed_ndarray = np.ndarray((inp_series.shape[0], inp_model.gensim_model.vector_size))
    
    for row, idx in enumerate(inp_series.index):
        
        tr_string = ''
        for wrd in inp_series[idx]:
            if wrd in inp_model.gensim_model.wv.vocab:
                tr_string = ' '.join([tr_string, wrd])

        transformed_ndarray[row] = np.mean(inp_model.transform(tr_string.split()), axis = 0)
    
    return transformed_ndarray


# In[68]:


get_ipython().run_cell_magic('time', '', 'warnings.filterwarnings("ignore")\n\n\nscores = [(\'model\', (0, 0, 0, 0), 0) for idx in range(5)]\nfor (sz, wind, it, mc) in tqdm(product(par_size, par_window, par_niter, par_min_count), \n                              total = len(list(product(par_size, par_window, par_niter, par_min_count)))):\n    model_x2v = W2VTransformer(size=sz, window = wind, hs = 1, iter = it, min_count = mc,\n                               seed = 16544873\n                              )\n    #model_x2v.fit(train)\n    model_x2v.fit(df.text_cl)\n    #train_x2v = transform_x2v(model_x2v, train)\n    train_x2v = np.nan_to_num(transform_x2v(model_x2v, train))\n\n    grid = GridSearchCV(pipe_cnt, parameters, cv = 5, scoring = \'roc_auc\', verbose = 0, n_jobs=-1)\n    ret = grid.fit(train_x2v, train_target)\n    if grid.best_score_ > scores[4][2]:\n        scores[4] = (grid.best_estimator_[\'clf\'].__class__, (sz, wind, it, mc), grid.best_score_)\n        scores.sort(key=lambda tup: tup[2], reverse = True)\n\n\n\n#warnings.filterwarnings("always")\nwarnings.filterwarnings("default")')


# Посмотрим на несколько лучших результатов

# In[69]:


scores

[(sklearn.svm._classes.LinearSVC, (200, 5, 50, 5), 0.8453186181634458),
 (sklearn.svm._classes.LinearSVC, (300, 5, 50, 5), 0.8447524810100875),
 (sklearn.svm._classes.LinearSVC, (200, 5, 40, 5), 0.8438951796456868),
 (sklearn.svm._classes.LinearSVC, (300, 5, 40, 5), 0.8374836539390291),
 (sklearn.svm._classes.LinearSVC, (300, 5, 50, 3), 0.8334031755938449)]
# Промежуточные выводы /параметры для W2VTransformer:
# - размер embedded вектора - 200
# - max расстояние между словами всегда 5
# - hs всегда 1
# - кол-во итерций построения словаря - 50
# - модель - LinearSVC
# 

# In[ ]:





# ### Попробуем определить причины ошибок по тесту

# In[70]:


model_x2v = W2VTransformer(size=200, window = 5, hs = 1, iter = 50, min_count = 5,
                           seed = 16544873
                          )
#model_x2v.fit(train)
model_x2v.fit(df.text_cl)
#train_x2v = transform_x2v(model_x2v, train)
train_x2v = np.nan_to_num(transform_x2v(model_x2v, train))

model = LinearSVC(max_iter = 2500)
model.fit(train_x2v, train_target)


# In[73]:


test_pred = model.predict(np.nan_to_num(transform_x2v(model_x2v, test)))


# In[74]:


confusion_matrix(test_target, test_pred)


# только половина отрицательных отзывов, посмотрим их

# In[75]:


for idx in range(test.shape[0]):
    if test_pred[idx] != test_target[test.index[idx]]:
        print(test.index[idx],test_target[test.index[idx]], df.loc[test.index[idx]].text)


# Если посмотреть на отзывы, в которых модель ошиблась - то в них нет ничего необычного.   
# Большая часть однозначно определяется как положительные или отрицательные.   
# Предполагаю, что влияют 2 фактора: несбалансированность класса и не больша выборка для обучения
# Попробуем исправить, настроив class_weights

# In[ ]:





# # Пробуем настроить параметры алгоритма

# #### в 3-4 итерации. в начале ищем приближенные лучшие значения(закомментированные параметры ниже),   
# #### потом их уточняем

# In[87]:


pipe_cnt = Pipeline([('clf', SVC()),
                      ])


# In[115]:


model_parameters = {
    'clf__max_iter' : [2500, 2000],# 1500],
    'clf__kernel': ['linear'],# 'poly', 'rbf', 'sigmoid',],
    #'clf__C': np.logspace(-2, 2, 10),
    'clf__C': np.linspace(1.7, 2, 40),
    #'clf__class_weight': [{0: 0.5 - shift, 1:0.5 + shift} for shift in np.linspace(-0.1, 0.1, 20)],# + ['balanced'],
    'clf__class_weight': [{0: 0.56 - shift, 1:0.44 + shift} for shift in np.linspace(-0.01, 0.01, 20)]# + ['balanced'],
}


# In[133]:


get_ipython().run_cell_magic('time', '', "\nmodel_x2v = W2VTransformer(size=200, window = 5, hs = 1, iter = 50, min_count = 5,\n                           seed = 16544873\n                          )\nmodel_x2v.fit(df.text_cl)\ntrain_x2v = np.nan_to_num(transform_x2v(model_x2v, train))\n\ngrid_tune = GridSearchCV(pipe_cnt, model_parameters, cv = 5, scoring = 'roc_auc', verbose = 1, n_jobs=-1)\ngrid_tune.fit(train_x2v, train_target)\ngrid_tune.best_estimator_")


# In[145]:


grid_tune.best_params_


# In[146]:


grid_tune.best_score_

0.842950668330993
# Посмотрим на несколько лучших результатов, после чего уточняем параметры и повторяем (несколько раз) GridSearchCV

# In[136]:


interested_tune = np.where(grid_tune.cv_results_['rank_test_score'] <= 5)
for field in model_parameters.keys():
    print(field)
    for idx in interested_tune:
        print(grid_tune.cv_results_['param_' + field][idx])


# In[137]:


grid_tune.cv_results_['mean_test_score'][np.where(grid_tune.cv_results_['rank_test_score'] <= 5)]

array([0.84258561, 0.84246868, 0.84295067, 0.84274984, 0.84246112])
# In[ ]:





# In[ ]:





# Остановимся на следующих параметрах:    
#     
# SVC
# kernel = 'linear'
# C =  2.41025641025641
# сlass_weight = {0: 0.5246153846153846, 1: 0.47538461538461535}
# max_iter = 2000

# Получим модель и submission для выбранных параметров

# In[ ]:





# ### Подготавливаем данные для обучения финальной модели и прогноза

# In[147]:


df_test = pd.read_csv(os.path.join(PATH_DATA, 'products_sentiment_test.tsv'),
                        index_col = None,
                        sep = '\t',
                     )
df_test.shape


# In[148]:


#df_test.head(), df_test.tail()


# In[149]:


subm = pd.read_csv(os.path.join(PATH_DATA, 'products_sentiment_sample_submission.csv'))
subm.shape


# In[150]:


#subm.head(), subm.tail()


# In[151]:


test_df = df_test.text.map(clean_text)
test_df = test_df.map(lem_text)
test_df = test_df.map(stem_text)
test_df = test_df.map(lambda x: x.split())


# In[152]:


test_x2v = np.nan_to_num(transform_x2v(model_x2v, test_df))


# In[ ]:





# ### Финальная модель для прогноза и прогноз

# In[ ]:





# In[153]:


pred = grid_tune.best_estimator_.predict(test_x2v)
subm.y = pred


# In[154]:


subm.to_csv(os.path.join(PATH_SUBM, 'gensim_w2v_own_lsvc_tuned.csv'), index = False)


# In[ ]:





# pb 0.74000

# In[ ]:




