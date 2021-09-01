#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# Подготовка окружения

# In[ ]:


import json
import zipfile
import os


# In[ ]:


get_ipython().system('pip install kaggle')


# In[ ]:


get_ipython().system('mkdir .kaggle')
get_ipython().system('touch .kaggle/kaggle.json')

api_token = {"username":"user","key":"api-key"}

with open('/content/.kaggle/kaggle.json', 'w') as file:
    json.dump(api_token, file)

get_ipython().system('chmod 600 /content/.kaggle/kaggle.json')
get_ipython().system('kaggle config path -p /content')


# In[ ]:


get_ipython().system('mv .kaggle /root/')


# In[ ]:


get_ipython().system('kaggle competitions download -c simplesentiment')


# In[ ]:





# In[ ]:


get_ipython().system('pip install fasttext')


# In[ ]:


import pandas as pd
import numpy as np
import pickle as pkl
import re


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


import fasttext


# In[ ]:


df = pd.read_csv('products_sentiment_train.tsv', 
                    header = None, 
                    index_col = None,
                    sep = '\t',
                   )
df.columns = ['text', 'target']
df.shape


# In[ ]:


clean_text = lambda x: re.sub(r"\s+", ' ', 
                              re.sub(r"[\d+]", '',
                                     re.sub(r"[^\w\s]", '', x.lower()).strip()
                                    )
                             )


# In[ ]:


df['text_cl'] = df.text.map(clean_text)
df['text_cl'] = df.text_cl.map(lambda x: x.split())


# In[ ]:





# In[ ]:


train, test, train_target, test_target = train_test_split(df.text_cl, df.target, 
                                                          test_size = 0.2, 
                                                          stratify = df.target, 
                                                          random_state = 52138,
                                                         )


# In[ ]:


# Трансфорование данных (отзыва) в вектор
# требуется исключить слова, которые не встречались, т.к. выдаст ошибку
# и полученные вектораотзыва усредняем по столбцам
def transform_x2v_wv(inp_wv, inp_series):
    transformed_ndarray = np.ndarray((inp_series.shape[0], inp_wv.get_dimension()))
    
    for row, idx in enumerate(inp_series.index):
        vect = []    
        for wrd in inp_series[idx]:
            #if wrd in inp_wv:
            vect.append(inp_wv.get_word_vector(wrd))

        transformed_ndarray[row] = np.mean(vect, axis = 0)
    
    return transformed_ndarray


# In[ ]:





# In[ ]:


# Трансфорование данных (отзыва) в вектор
# требуется исключить слова, которые не встречались, т.к. выдаст ошибку
# и полученные вектораотзыва усредняем по столбцам
def transform_x2v_model(inp_model, inp_series):
    transformed_ndarray = np.ndarray((inp_series.shape[0], inp_model.dim()))
    
    for row, idx in enumerate(inp_series.index):
        vect = []    
        for wrd in inp_series[idx]:
            #if wrd in inp_wv:
            vect.append(inp_model.get_word_vector(wrd))

        transformed_ndarray[row] = np.mean(vect, axis = 0)
    
    return transformed_ndarray


# In[ ]:


pipe_cnt = Pipeline([#('w2v', W2VTransformer(size=100, min_count=1, seed=1)),
                     ('clf', LogisticRegression()),
                      ])

parameters = [
            {'clf': [LogisticRegression(max_iter = 150)]}, 
            {'clf': [LinearSVC(max_iter = 1500)]},
            {'clf': [SGDClassifier()]}, 
            {'clf': [RandomForestClassifier()]},
            #{'clf': [xgb.XGBClassifier(eval_metric = 'auc', use_label_encoder=False)]},
                     #xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0),
            #{'clf': [LGBMClassifier()]},
]


# In[ ]:



#!wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/amazon_review_polarity.bin
#!wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/amazon_review_full.bin

#!wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/yelp_review_polarity.bin
#!wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/yelp_review_full.bin


# In[ ]:


get_ipython().system('wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz')
get_ipython().system('gunzip cc.en.300.bin.gz')


# In[ ]:


#!ls -la


# In[ ]:


#model = fasttext.load_model("amazon_review_polarity.bin")
#model = fasttext.load_model("amazon_review_full.bin")
#model = fasttext.load_model("yelp_review_polarity.bin")
#model = fasttext.load_model("yelp_review_full.bin")
model = fasttext.load_model("cc.en.300.bin")
model.get_dimension()


# In[ ]:


#$dir(model)


# In[ ]:


#train_x2v = np.nan_to_num(transform_x2v_kv(model, train))
train_x2v = np.nan_to_num(transform_x2v_wv(model, train))

grid = GridSearchCV(pipe_cnt, parameters, cv = 5, scoring = 'roc_auc', verbose = 1, n_jobs=-1)
ret = grid.fit(train_x2v, train_target)


# In[ ]:


grid.best_score_, grid.best_estimator_['clf'].__class__


# In[ ]:


[(el0, el1) for el0, el1 in zip(grid.cv_results_['mean_test_score'], grid.cv_results_['params'])]


# amazon_review_polarity   
# (0.7970514863770441, sklearn.svm._classes.LinearSVC)   
# amazon_review_full   
# (0.8305306649627135, sklearn.linear_model._stochastic_gradient.SGDClassifier)   
# yelp_review_polarity   
# (0.7924044065017697, sklearn.linear_model._stochastic_gradient.SGDClassifier)   
# yelp_review_full   
# (0.7927665893943784, sklearn.linear_model._stochastic_gradient.SGDClassifier)   
# 
# 

# In[ ]:


pipe_model = Pipeline([('clf', SGDClassifier()),
])

parameters_model = {'clf__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
              'clf__penalty': ['l1', 'l2'],
              'clf__class_weight': [{0: 0.5 - shift, 1:0.5 + shift} for shift in np.linspace(-0.01, 0.01, 40)] + ['balanced'],
              }


# In[ ]:


grid_tune = GridSearchCV(pipe_model, parameters_model, cv = 5, scoring = 'roc_auc', verbose = 1, n_jobs=-1)
grid_tune.fit(train_x2v, train_target)
grid_tune.best_estimator_


# In[ ]:


grid_tune.best_score_


# In[ ]:





# In[ ]:





# In[ ]:


train_ft = np.nan_to_num(transform_x2v_wv(model, df['text_cl']))


# In[ ]:


#model_tuned = SGDClassifier(class_weight={0: 0.5089743589743589,
#                                          1: 0.491025641025641},
#                            learning_rate='optimal', loss='modified_huber',
#                            penalty='l2',
#                           )


# In[ ]:


#model_tuned.fit(train_ft, df.target)


# In[ ]:


#pred_train_ft = model_tuned.predict(train_ft)


# In[ ]:


pred_train_ft = grid_tune.best_estimator_.predict(train_ft)


# In[ ]:


roc_auc_score(df.target, pred_train_ft)


# In[ ]:


confusion_matrix(df.target, pred_train_ft)


# In[ ]:


with open('pred_train_ft.pkl', 'wb') as fd:
    pkl.dump(pred_train_ft, fd)


# In[ ]:





# # Предсказания для submit

# In[ ]:


subm = pd.read_csv('products_sentiment_sample_submission.csv')
subm.shape


# In[ ]:


df_test = pd.read_csv('products_sentiment_test.tsv',
                        index_col = None,
                        sep = '\t',
                     )
df_test.shape


# In[ ]:


df_test['text_cl'] = df_test.text.map(clean_text)
df_test['text_cl'] = df_test.text_cl.map(lambda x: x.split())


# In[ ]:


test_x2v = np.nan_to_num(transform_x2v_wv(model, df_test['text_cl']))


# In[ ]:


pred_test_ft = grid_tune.best_estimator_.predict(test_x2v)
#pred_test_ft = model_tuned.predict(test_x2v)


# In[ ]:


subm.y = pred_test_ft


# In[ ]:


subm.to_csv('ft_ccen300_nols.csv', index=False)


# In[ ]:


with open('pred_test_ft.pkl', 'wb') as fd:
    pkl.dump(pred_test_ft, fd)


# In[ ]:





# In[ ]:





# pb 0.78666
