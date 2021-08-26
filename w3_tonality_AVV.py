#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Kagle inclass https://www.kaggle.com/c/simplesentiment/overview

# In[ ]:


import os
from pathlib import Path

import pandas as pd
import numpy as np


# In[ ]:


import nltk
#nltk.download('movie_reviews')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('rslp')

import nltk.stem as st

#import re


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score 

#from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


# In[ ]:


#import xgboost as xgb


# In[ ]:





# In[ ]:


PATH_DATA = os.path.join(Path.cwd(), 'data')
PATH_SUBM = os.path.join(Path.cwd(), 'submissions')


# In[ ]:





# In[ ]:





# In[ ]:


df = pd.read_csv(os.path.join(PATH_DATA, 'products_sentiment_train.tsv'), 
                    header = None, 
                    index_col = None,
                    sep = '\t',
                   )
df.columns = ['text', 'target']
df.shape


# In[ ]:


df.head()


# In[ ]:


df.target.value_counts()


# Классы явно не сбалансированы.   
# Т.к. мы работаем с текстами есть следующие варианты:   
#     1. добавить негативные примеры. + увеличение выборки, баланс классов. - необходимо найти размеченную, подходящую под тематику, выборку   
#     2. дублировать некоторые строки, например те, в которых ошибается модель   
#     3. настроить параметр class_weights
#     

# ### Приводим к строчным (на всякий) и очищаем по порядку:

# In[ ]:


# - все спецсимволы
# - все цифры
# - заменяем множественные пробелы на единичные


# In[ ]:


clean_text = lambda x: re.sub(r"\s+", ' ', 
                              re.sub(r"[\d+]", '',
                                     re.sub(r"[^\w\s]", '', x.lower()).strip()
                                    )
                             )


# In[ ]:


df['text_cl'] = df.text.map(clean_text)


# In[ ]:





# In[ ]:





# In[ ]:


#lem_text("don't"), lem_text("i'll")


# In[ ]:


#stem_text("don't"), stem_text("i'll")


# ### Лемматизация & стемминг

# In[ ]:


lemm = st.WordNetLemmatizer()
lem_text = lambda x: ' '.join([lemm.lemmatize(el) for el in x.split()])


#stemm = st.PorterStemmer()
#stemm = st.Cistem()
#stemm = st.LancasterStemmer()

#stemm = st.ISRIStemmer()
stemm = st.RSLPStemmer()
stem_text = lambda x: ' '.join([stemm.stem(el) for el in x.split()])


# In[ ]:


df['text_cl'] = df.text.map(clean_text)
df['text_cl'] = df.text_cl.map(lem_text)
df['text_cl'] = df.text_cl.map(stem_text)


# In[ ]:


df.head()


# ### Подготавливаем к обучению по сетке

# In[ ]:


train, test, train_target, test_target = train_test_split(df.text_cl, df.target, 
                                                          test_size = 0.2, 
                                                          stratify = df.target, 
                                                          random_state = 52138,
                                                         )


# ### параметры для перебора

# In[ ]:


pipe_cnt = Pipeline([('vect', CountVectorizer()), 
                     ('clf', SGDClassifier()),
                      ])


# In[ ]:


parameters = [
    {
        'vect': [CountVectorizer(), TfidfVectorizer()],
        #'vect__analyzer': ['word','char_wb'],
        'vect__analyzer': ['char_wb'],
        #'vect__ngram_range': [(1, 1), (1, 2), (2, 2), (2, 3), (2, 4), (3, 4), (3, 5), (2, 5)],
        'vect__ngram_range': [(2, 4), (3, 4), (3, 5), (2, 5)],
        'vect__stop_words': [None, nltk.corpus.stopwords.words('english')],
        'vect__max_df': [0.75, 1.0],
        'vect__min_df': [10, 1,],# 50],
        'clf': [LogisticRegression(), 
                LinearSVC(),
                SGDClassifier(), 
                RandomForestClassifier()
                #xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0),
                #xgb.XGBClassifier(eval_metric = 'auc'),
               ],
    },
    
]


# In[ ]:


grid = GridSearchCV(pipe_cnt, parameters, cv = 5, scoring = 'roc_auc', verbose = 2, n_jobs=-1)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'grid.fit(train, train_target)\ngrid.best_estimator_')


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_score_


# In[ ]:


grid.cv_results_.keys()


# Посмотрим на несколько лучших результатов

# In[ ]:


#grid.cv_results_['rank_test_score']


# In[ ]:


interested = np.where(grid.cv_results_['rank_test_score'] <= 5)
for field in parameters[0].keys():
    print(field)
    for idx in interested:
        print(grid.cv_results_['param_' + field][idx])


# Промежуточные выводы:
# - токенизация всегда TfidfVectorizer
# - метод разбиения - char_wb
# - ngram_range - (3, 4) для 1х и 3х результатов, (3,5) для 5х.
#    включу ngram_range в перебор при настройке параметров
# - модель - LogisticRegression
# - min_df - всегда 1
# - max_df - 0.75 для 3х результатов, 1.0 для 1х и 5х. 
#    включу max_df в перебор при настройке параметров
# - стоп слова не влияют никаким образом (что очень удивительно)
# 

# In[ ]:





# ### Попробуем определить причины ошибок по тесту

# In[ ]:


vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), 
                             max_df=1.0, min_df=1, 
                             stop_words=None
                            )
vectorizer.fit(train)


# In[ ]:


model = LogisticRegression(
)
model.fit(vectorizer.transform(train), train_target)


# In[ ]:


test_pred = model.predict(vectorizer.transform(test))


# In[ ]:


confusion_matrix(test_target, test_pred)


# только половина отрицательных отзывов, посмотрим их

# In[ ]:


for idx in range(test.shape[0]):
    if test_pred[idx] != test_target[test.index[idx]]:
        print(test.index[idx],test_target[test.index[idx]], df.loc[test.index[idx]].text)


# Если посмотреть на отзывы, в которых модель ошиблась - то в них нет ничего необычного.   
# Большая часть однозначно определяется как положителбные или отрицательные.   
# Предполагаю, что влияют 2 фактора: несбалансированность класса и предел возможностей tf-idf   
# Попробуем исправить, настроив class_weights

# In[ ]:





# # Пробуем настроить параметры алгоритма

# #### в 3-4 итерации. в начале ищем приближенные лучшие значения(закомментированные параметры ниже),   
# #### потом их уточняем

# In[ ]:


pipe_cnt = Pipeline([('vect', TfidfVectorizer(analyzer='char_wb')), 
                     ('clf', LogisticRegression()),
                      ])


# In[ ]:


model_parameters = {
    'vect__ngram_range': [(3, 4), ], #(3, 5)],
    #'vect__max_df': [0.75, 0.65, 0.85,], #1.0],
    'vect__max_df': [0.75],
    'clf__solver': ['liblinear',], #'lbfgs'],
    'clf__penalty': ['l2',], #'l1'], 
    #'clf__C': np.logspace(1, 5, 10),
    'clf__C': np.linspace(1.69, 1.74, 20),
    #'clf__class_weight': [{0: 0.5 - shift, 1:0.5 + shift} for shift in np.linspace(-0.1, 0.1, 20)],# + ['balanced'],
    'clf__class_weight': [{0: 0.59 - shift, 1:0.41 + shift} for shift in np.linspace(-0.001, 0.001, 50)],# + ['balanced'],
    'clf__max_iter': [50, 75],# 100, 200],
}


# In[ ]:


grid_tune = GridSearchCV(pipe_cnt, model_parameters, cv = 5, scoring = 'roc_auc', verbose = 2, n_jobs=-1)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'grid_tune.fit(train, train_target)\ngrid_tune.best_estimator_')


# In[ ]:


grid_tune.best_params_


# In[ ]:


grid_tune.best_score_


# Посмотрим на несколько лучших результатов, после чего уточняем параметры и повторяем (несколько раз) GridSearchCV

# In[ ]:


interested_tune = np.where(grid_tune.cv_results_['rank_test_score'] <= 5)
for field in model_parameters.keys():
    print(field)
    for idx in interested_tune:
        print(grid_tune.cv_results_['param_' + field][idx])


# In[ ]:


grid_tune.cv_results_['mean_test_score'][np.where(grid_tune.cv_results_['rank_test_score'] <= 5)]


# Остановимся на следующих параметрах:    
# TfidfVectorizer    
# analyzer ='char_wb'    
# max_df = 0.75    
# ngram_range = (3, 4)    
#     
# LogisticRegression   
# C =  1.7189473684210526   
# сlass_weight = {0: 0.5894081632653061, 1: 0.41059183673469385}   
# max_iter = 50 (возможно, мало итераций т.к. обучались не на всех данных, так что для submission можно попробовать и 75)   
# penalty = 'l2'   
# solver = 'liblinear'   
#  

# Получим модель и submission для выбранных параметров

# In[ ]:





# ### Подготавливаем данные для обучения финальной модели и прогноза

# In[ ]:


df_test = pd.read_csv(os.path.join(PATH_DATA, 'products_sentiment_test.tsv'),
                        index_col = None,
                        sep = '\t',
                     )
df_test.shape


# In[ ]:


#df_test.head(), df_test.tail()


# In[ ]:


subm = pd.read_csv(os.path.join(PATH_DATA, 'products_sentiment_sample_submission.csv'))
subm.shape


# In[ ]:


#subm.head(), subm.tail()


# In[ ]:


vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), 
                             max_df=0.75, min_df=1, 
                             stop_words=None
                            )
vectorizer.fit(df.text_cl)
train_df = vectorizer.transform(df.text_cl)


# In[ ]:


test_df = df_test.text.map(clean_text)
test_df = test_df.map(lem_text)
test_df = test_df.map(stem_text)
test_df = vectorizer.transform(test_df)


# In[ ]:





# ### Финальная модель для прогноза и прогноз

# In[ ]:


model = LogisticRegression(solver = 'liblinear',   
                           penalty = 'l2',
                           C = 1.7189473684210526,
                           class_weight = {0: 0.5894081632653061, 1: 0.41059183673469385},
                           max_iter = 75, #75
)


# In[ ]:


model.fit(train_df, df.target)


# In[ ]:


pred = model.predict(test_df)
subm.y = pred


# In[ ]:


subm.to_csv(os.path.join(PATH_SUBM, 'tf_st_lm_lr_tuned_75.csv'), index = False)


# In[ ]:





# pb 0.79333

# In[ ]:




