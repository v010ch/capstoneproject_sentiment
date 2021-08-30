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

#from gensim.models.doc2vec import Doc2Vec, TaggedDocument
#from gensim.sklearn_api import W2VTransformer
import gensim.downloader as api


# In[4]:


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


# In[5]:


import xgboost as xgb
from lightgbm import LGBMClassifier


# In[ ]:





# In[6]:


PATH_DATA = os.path.join(Path.cwd(), 'data')
PATH_SUBM = os.path.join(Path.cwd(), 'submissions')


# In[ ]:





# In[ ]:





# In[31]:


df = pd.read_csv(os.path.join(PATH_DATA, 'products_sentiment_train.tsv'), 
                    header = None, 
                    index_col = None,
                    sep = '\t',
                   )
df.columns = ['text', 'target']
df.shape


# In[32]:


df.head()


# In[33]:


df.target.value_counts()


# Классы явно не сбалансированы.   
# Т.к. мы работаем с текстами есть следующие варианты:   
#     1. добавить негативные примеры. + увеличение выборки, баланс классов. - необходимо найти размеченную, подходящую под тематику, выборку   
#     2. дублировать некоторые строки, например те, в которых ошибается модель   
#     3. настроить параметр class_weights
#     

# ### Приводим к строчным (на всякий) и очищаем по порядку:

# In[34]:


# - все спецсимволы
# - все цифры
# - заменяем множественные пробелы на единичные


# In[35]:


clean_text = lambda x: re.sub(r"\s+", ' ', 
                              re.sub(r"[\d+]", '',
                                     re.sub(r"[^\w\s]", '', x.lower()).strip()
                                    )
                             )


# In[36]:


df['text_cl'] = df.text.map(clean_text)


# In[ ]:





# In[37]:


#tagged_data[:5]


# In[38]:


#lem_text("don't"), lem_text("i'll")


# In[39]:


#stem_text("don't"), stem_text("i'll")


# ### Лемматизация & стемминг

# In[40]:


lemm = st.WordNetLemmatizer()
lem_text = lambda x: ' '.join([lemm.lemmatize(el) for el in x.split()])


stemm = st.RSLPStemmer()
stem_text = lambda x: ' '.join([stemm.stem(el) for el in x.split()])


# In[41]:


df['text_cl'] = df.text.map(clean_text)
#df['text_cl'] = df.text_cl.map(lem_text)
#df['text_cl'] = df.text_cl.map(stem_text)


# In[42]:


df['text_cl'] = df.text_cl.map(lambda x: x.split())


# In[43]:


df.head()


# In[ ]:





# In[ ]:





# Загружаем готовый wordvector

# https://github.com/RaRe-Technologies/gensim-data

# In[126]:


#word_vectors = api.load("glove-wiki-gigaword-300")
#word_vectors = api.load("glove-twitter-200")#, return_path = True
word_vectors = api.load("word2vec-google-news-300")


# In[127]:


word_vectors


# 'C:\\Users\\****/gensim-data\\glove-wiki-gigaword-300\\glove-wiki-gigaword-300.gz'

# In[ ]:





# ### Подготавливаем к обучению по сетке

# In[107]:


train, test, train_target, test_target = train_test_split(df.text_cl, df.target, 
                                                          test_size = 0.2, 
                                                          stratify = df.target, 
                                                          random_state = 52138,
                                                         )


# In[ ]:





# ### Параметры для перебора

# In[108]:


pipe_cnt = Pipeline([#('w2v', W2VTransformer(size=100, min_count=1, seed=1)),
                     ('clf', LogisticRegression()),
                      ])


# In[109]:


parameters_v0 = {
        'clf': [#LogisticRegression(), 
                LinearSVC(),
                #SGDClassifier(), 
                #RandomForestClassifier(),
                #xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0),
                #xgb.XGBClassifier(eval_metric = 'auc'),
               ],
    }


# In[116]:


parameters = [
            {'clf': [LogisticRegression(max_iter = 150)]}, 
            {'clf': [LinearSVC(max_iter = 1500)]},
            {'clf': [SGDClassifier()]}, 
            {'clf': [RandomForestClassifier()]},
            {'clf': [xgb.XGBClassifier(eval_metric = 'auc', use_label_encoder=False)]},
                     #xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0),
            {'clf': [LGBMClassifier()]},
]


# In[ ]:





# In[117]:


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


# In[118]:


# Трансфорование данных (отзыва) в вектор
# требуется исключить слова, которые не встречались, т.к. выдаст ошибку
# и полученные вектораотзыва усредняем по столбцам
def transform_x2v_kv(inp_wv, inp_series):
    transformed_ndarray = np.ndarray((inp_series.shape[0], inp_wv.vector_size))
    
    for row, idx in enumerate(inp_series.index):
        vect = []    
        for wrd in inp_series[idx]:
            if wrd in inp_wv:
                vect.append(inp_wv.get_vector(wrd))

        transformed_ndarray[row] = np.mean(vect, axis = 0)
    
    return transformed_ndarray


# In[131]:


get_ipython().run_cell_magic('time', '', '#warnings.filterwarnings("ignore")\n\ntrain_x2v = np.nan_to_num(transform_x2v_kv(word_vectors, train))\n\ngrid = GridSearchCV(pipe_cnt, parameters, cv = 5, scoring = \'roc_auc\', verbose = 1, n_jobs=-1)\nret = grid.fit(train_x2v, train_target)\n\n\n#warnings.filterwarnings("always")\n#warnings.filterwarnings("default")')


# Посмотрим на несколько лучших результатов

# In[132]:


grid.best_score_, grid.best_estimator_['clf'].__class__


# In[133]:


[(el0, el1) for el0, el1 in zip(grid.cv_results_['mean_test_score'], grid.cv_results_['params'])]


# In[ ]:




word2vec-google-news-300

[(0.8625631242771201, {'clf': LogisticRegression(max_iter=150)}),
 (0.8636084672595828, {'clf': LinearSVC(max_iter=1500)}),
 (0.858903773711076, {'clf': SGDClassifier()}),
 (0.8193513369096534, {'clf': RandomForestClassifier()}),
 (0.8438609086428558,
  {'clf': XGBClassifier(base_score=None, booster=None, colsample_bylevel=None,
                 colsample_bynode=None, colsample_bytree=None, eval_metric='auc',
                 gamma=None, gpu_id=None, importance_type='gain',
                 interaction_constraints=None, learning_rate=None,
                 max_delta_step=None, max_depth=None, min_child_weight=None,
                 missing=nan, monotone_constraints=None, n_estimators=100,
                 n_jobs=None, num_parallel_tree=None, random_state=None,
                 reg_alpha=None, reg_lambda=None, scale_pos_weight=None,
                 subsample=None, tree_method=None, use_label_encoder=False,
                 validate_parameters=None, verbosity=None)}),
 (0.8457348235492252, {'clf': LGBMClassifier()})]glove-twitter-200

[(0.8583412586962285, {'clf': LogisticRegression(max_iter=150)}),
 (0.8469416148422233, {'clf': LinearSVC(max_iter=1500)}),
 (0.847100122100122, {'clf': SGDClassifier()}),
 (0.8061515366535652, {'clf': RandomForestClassifier()}),
 (0.829008004616524,
  {'clf': XGBClassifier(base_score=None, booster=None, colsample_bylevel=None,
                 colsample_bynode=None, colsample_bytree=None, eval_metric='auc',
                 gamma=None, gpu_id=None, importance_type='gain',
                 interaction_constraints=None, learning_rate=None,
                 max_delta_step=None, max_depth=None, min_child_weight=None,
                 missing=nan, monotone_constraints=None, n_estimators=100,
                 n_jobs=None, num_parallel_tree=None, random_state=None,
                 reg_alpha=None, reg_lambda=None, scale_pos_weight=None,
                 subsample=None, tree_method=None, use_label_encoder=False,
                 validate_parameters=None, verbosity=None)}),
 (0.8343619897118882, {'clf': LGBMClassifier()})]glove-wiki-gigaword-300

[(0.855325131821075, {'clf': LogisticRegression()}),
 (0.8418739025229897, {'clf': LinearSVC(max_iter=1500)}),
 (0.838749947370637, {'clf': SGDClassifier()}),
 (0.802623171284429, {'clf': RandomForestClassifier()}),
 (0.8253392891197151,
  {'clf': XGBClassifier(base_score=None, booster=None, colsample_bylevel=None,
                 colsample_bynode=None, colsample_bytree=None, eval_metric='auc',
                 gamma=None, gpu_id=None, importance_type='gain',
                 interaction_constraints=None, learning_rate=None,
                 max_delta_step=None, max_depth=None, min_child_weight=None,
                 missing=nan, monotone_constraints=None, n_estimators=100,
                 n_jobs=None, num_parallel_tree=None, random_state=None,
                 reg_alpha=None, reg_lambda=None, scale_pos_weight=None,
                 subsample=None, tree_method=None, use_label_encoder=False,
                 validate_parameters=None, verbosity=None)}),
 (0.8239905688181549, {'clf': LGBMClassifier()})]
# In[ ]:





# In[ ]:





# 

# In[ ]:





# ### Попробуем определить причины ошибок по тесту

# In[139]:


#train_x2v = np.nan_to_num(transform_x2v(model_x2v, train))

model = LinearSVC(max_iter = 2500)
model.fit(train_x2v, train_target)


# In[142]:


test_pred = model.predict(np.nan_to_num(transform_x2v_kv(word_vectors, test)))


# In[143]:


confusion_matrix(test_target, test_pred)


# только половина отрицательных отзывов, посмотрим их

# In[144]:


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

# In[145]:


pipe_cnt = Pipeline([('clf', SVC()),
                      ])


# In[168]:


model_parameters = {
    'clf__max_iter' : [2500, 2000],# 1500],
    'clf__kernel': ['rbf', 'linear'], #'poly', 'sigmoid',],
    #'clf__C': np.logspace(-1, 2, 20),
    'clf__C': np.linspace(3.5, 4, 10),
    'clf__class_weight': [{0: 0.53 - shift, 1:0.47 + shift} for shift in np.linspace(-0.01, 0.01, 20)] + ['balanced'],
    #'clf__class_weight': [{0: 0.56 - shift, 1:0.44 + shift} for shift in np.linspace(-0.01, 0.01, 20)]# + ['balanced'],
}


# In[ ]:





# In[169]:


get_ipython().run_cell_magic('time', '', "\ntrain_x2v = np.nan_to_num(transform_x2v_kv(word_vectors, train))\n\ngrid_tune = GridSearchCV(pipe_cnt, model_parameters, cv = 5, scoring = 'roc_auc', verbose = 1, n_jobs=-1)\ngrid_tune.fit(train_x2v, train_target)\ngrid_tune.best_estimator_")


# In[170]:


grid_tune.best_params_


# In[171]:


grid_tune.best_score_


# 0.8729543895365397
# {'clf__C': 3.6666666666666665,
#  'clf__class_weight': {0: 0.531578947368421, 1: 0.4684210526315789},
#  'clf__kernel': 'rbf',
#  'clf__max_iter': 2500}
# 
# 0.8729120383785698
# {'clf__C': 3.7368421052631575,
#  'clf__class_weight': {0: 0.5342105263157895, 1: 0.46578947368421053},
#  'clf__kernel': 'rbf',
#  'clf__max_iter': 2500}
#  
# 0.8729205209935433
# ('clf', SVC(C=3.79269019073225,
# class_weight={0: 0.5368421052631579, : 0.4631578947368421},
# max_iter=2500))

# Посмотрим на несколько лучших результатов, после чего уточняем параметры и повторяем (несколько раз) GridSearchCV

# In[173]:


interested_tune = np.where(grid_tune.cv_results_['rank_test_score'] <= 5)
for field in model_parameters.keys():
    print(field)
    for idx in interested_tune:
        print(grid_tune.cv_results_['param_' + field][idx])


# In[174]:


grid_tune.cv_results_['mean_test_score'][np.where(grid_tune.cv_results_['rank_test_score'] <= 5)]


# In[ ]:





# In[ ]:





# Остановимся на следующих параметрах:    
# 
# word_vectors - "word2vec-google-news-300"   
# SVC
# kernel = 'rbf'
# C =  3.6666666666666665
# сlass_weight = {0: 0.531578947368421, 1: 0.4684210526315789}
# max_iter = 2500

# Получим модель и submission для выбранных параметров

# In[ ]:





# ### Подготавливаем данные для обучения финальной модели и прогноза

# In[175]:


df_test = pd.read_csv(os.path.join(PATH_DATA, 'products_sentiment_test.tsv'),
                        index_col = None,
                        sep = '\t',
                     )
df_test.shape


# In[176]:


#df_test.head(), df_test.tail()


# In[177]:


subm = pd.read_csv(os.path.join(PATH_DATA, 'products_sentiment_sample_submission.csv'))
subm.shape


# In[178]:


#subm.head(), subm.tail()


# In[179]:


test_df = df_test.text.map(clean_text)
#test_df = test_df.map(lem_text)
#test_df = test_df.map(stem_text)
test_df = test_df.map(lambda x: x.split())


# In[180]:


test_x2v = np.nan_to_num(transform_x2v_kv(word_vectors, test_df))


# In[ ]:





# ### Финальная модель для прогноза и прогноз

# In[ ]:





# In[181]:


pred = grid_tune.best_estimator_.predict(test_x2v)
subm.y = pred


# In[183]:


subm.to_csv(os.path.join(PATH_SUBM, 'gensim_w2v_gglnews300_svc_tuned_no_ls.csv'), index = False)


# In[ ]:





# pb 0.80888

# In[ ]:




