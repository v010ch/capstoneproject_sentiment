#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Kagle inclass https://www.kaggle.com/c/simplesentiment/overview

# Работа выполнена в colab.   
# Состоит из 2-х частей  
# - обучение модели
# - предсказание submission
# 
# При предсказании, по неустановленной на данный момент причине, модель не справляется с объемом 500 строк и падает по памяти. Поэтому данная часть повторяется несколько раз: загрузка модели, обработка части строк, сохранение результата. После обработки всех строк результат объединяется.   
# Для прода такой подхож не подходит, а для kaggle сгодится.

# ОБщая часть: загрузка библиотек, данных. Настройка окружения kaggle.

# In[ ]:


import os
from pathlib import Path
import json

import pandas as pd
import numpy as np
import pickle as pkl

import re
import random


# In[ ]:


#from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#from sklearn.linear_model import LogisticRegression, SGDClassifier
#from sklearn.svm import LinearSVC
#from sklearn.svm import SVC
#from sklearn.ensemble import RandomForestClassifier

#from sklearn.model_selection import GridSearchCV, cross_val_score 

#from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


# In[ ]:


import torch


# In[ ]:


import transformers
from transformers import BertModel 
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments

from transformers.file_utils import is_tf_available, is_torch_available
from transformers import pipeline


# In[ ]:


from transformers import BertForSequenceClassification


# In[ ]:





# In[ ]:


PATH_DATA = os.path.join(Path.cwd(), 'data')
PATH_SUBM = os.path.join(Path.cwd(), 'submissions')


# In[ ]:





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


# Получаем верные метки. Для второй части нас здесь интересует df.target, seed и лямбда функция clean_text.

# In[ ]:


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(seed)


# In[ ]:


set_seed(522)


# In[ ]:


df = pd.read_csv(os.path.join('./', 'products_sentiment_train.tsv'), 
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


# При предсказании можно переходить ко второй части.

# Часть 1 (продолжение). Построение модели.

# In[ ]:


df['text_cl'] = df.text.map(clean_text)


# In[ ]:


train, test, train_target, test_target = train_test_split(df.text, df.target, 
                                                          test_size = 0.2, 
                                                          stratify = df.target, 
                                                          random_state = 52138,
                                                         )


# In[ ]:





# Устанавливаем transformers, если не установлен.

# In[ ]:


get_ipython().system('pip install transformers')


# In[ ]:





# Загружаем выбраннную для transfer learning модель.

# In[ ]:


PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
#PRE_TRAINED_MODEL_NAME = "siebert//sentiment-roberta-large-english"
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


# In[ ]:


max(df.text_cl.map(lambda x: len(x.split()))), max(df.text.map(lambda x: len(x.split())))


# в выбранной для обучения части датасета максимальная длинна отзыва - 92 слова. возьмем максимальную длинну - 96, кратную 8.

# In[ ]:


max_length = 96


# In[ ]:


#tokenizer(['hello people twinky', 'world'], truncation=True, padding=True, max_length=max_length)


# In[ ]:


#tokenizer(['hello', 'world'], truncation=True, padding=True, max_length=max_length)


# In[ ]:


train_tokens = tokenizer(list(train.values), truncation=True, padding=True, max_length=max_length)


# In[ ]:


test_tokens = tokenizer(list(test.values), truncation=True, padding=True, max_length=max_length)


# В модель необходимо передать данные в виде torch tensor. Подготавливаем.

# In[ ]:


class TonalityDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


# In[ ]:


train_dataset = TonalityDataset(train_tokens, train_target.values)
test_dataset  = TonalityDataset(test_tokens,  test_target.values)


# In[ ]:





# Дообучаем модель на наших данных.

# In[ ]:


model = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, 
                                                      num_labels = df.target.value_counts().shape[0])#.to("cuda")


# In[ ]:


training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=20,   # batch size for evaluation
    warmup_steps=400,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    logging_steps=50,               # log & save weights each logging_steps
    evaluation_strategy="steps",     # evaluate each `logging_steps`
)


# In[ ]:


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    rocauc = roc_auc_score(labels, preds)
    return {
        'roc-auc': rocauc,
    }


# In[ ]:


trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,          # evaluation dataset
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
)


# In[ ]:


trainer.train()


# In[ ]:


trainer.evaluate()


# In[ ]:


model_path = "capstone_tonality_bert_base_cased_v1"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)


# In[ ]:





# In[ ]:





# In[ ]:





# Часть 2. Подготовка прогноза.

# загружаем токенизатор/модель.

# In[ ]:


model_path = "capstone_tonality_bert_base_cased_v1"


# In[ ]:


tokenizer = BertTokenizer.from_pretrained(model_path)


# In[ ]:


model = BertForSequenceClassification.from_pretrained(model_path, 
                                                      num_labels = df.target.value_counts().shape[0])


# In[ ]:


#dir(model)


# In[ ]:





# Данные для прогноза.

# In[ ]:


df_subm = pd.read_csv(os.path.join('./', 'products_sentiment_test.tsv'),
                        index_col = None,
                        sep = '\t',
                     )
df_subm.shape


# In[ ]:


df_subm['text_cl'] = df_subm.text.map(clean_text)


# In[ ]:


max_length = 96


# In[ ]:


#%%time
#print(tt)
#outputs = list()
#for idx in range(5):
#  print(idx)
#  subm_tokens = tokenizer(list(df_subm['text_cl'].values[idx*100:(idx+1)*100]), 
#                        truncation=True, padding=True, max_length=max_length, 
#                        return_tensors="pt"
#                        )
#  outputs.append(model(**subm_tokens))


# Подмассив задаем через idx = 0..4

# In[ ]:


get_ipython().run_cell_magic('time', '', '#idx = 0..4\nidx = 4\nsubm_tokens = tokenizer(list(df_subm[\'text_cl\'].values[idx*100:(idx + 1)*100]), \n                        truncation=True, padding=True, max_length=max_length, \n                        return_tensors="pt"\n                        )')


# In[ ]:


#subm_tokens


# Предсказываем для выбранного подмассива. Переводим в вероятности. Выбираем класс по максимому.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'outp = model(**subm_tokens)')


# In[ ]:


#type(outp_p1)


# In[ ]:


#outp


# In[ ]:


outp = outp[0].softmax(1)


# Сохраняем не класс, а обе вероятности. На случай если будем настраивать порог.

# In[ ]:


with open(os.path.join('./', f'outp_p{idx}.pkl'), 'wb') as fd:
    pkl.dump(outp, fd)


# In[ ]:


#outp


# Загружаем все подготовленные веростности.

# In[ ]:


submit_y = list()
for idx in range(5):
    with open(os.path.join('./', f'outp_p{idx}.pkl'), 'rb') as fd:
        outp = pkl.load(fd)
  
      submit_y += list(map(lambda x: x.argmax().item(), outp))


# In[ ]:


len(submit_y)


# In[ ]:


#outp_p1[0].softmax(1)[5].argmax().item()


# In[ ]:


subm = pd.read_csv(os.path.join('./', 'products_sentiment_sample_submission.csv'))
subm.shape


# In[ ]:


subm.y = submit_y


# In[ ]:


subm.to_csv(os.path.join('./', 'bert_base_cased.csv'), index = False)


# In[ ]:


get_ipython().system('ls')


# In[ ]:





# In[ ]:





# In[ ]:


from google.colab import files
files.download('bert_base_cased.csv') 


# pb 0.85111

# In[ ]:




