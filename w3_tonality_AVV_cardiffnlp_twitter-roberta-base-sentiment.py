#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Kagle inclass https://www.kaggle.com/c/simplesentiment/overview

# Работа выполнена в colab.   
# 
# При предсказании, по неустановленной на данный момент причине, модель не справляется с объемом 500 строк и падает по памяти. Поэтому данная часть повторяется несколько раз: загрузка модели, обработка части строк, сохранение результата. После обработки всех строк результат объединяется.   
# Для прода такой подхож не подходит, а для kaggle сгодится. Хотя, полагаю в проде оценивается по 1 отзыву за раз.

# ОБщая часть: загрузка библиотек, данных. Настройка окружения kaggle.

# In[1]:


import os
from pathlib import Path
import json

import pandas as pd
import numpy as np
import pickle as pkl

import re
import random


# In[3]:


import torch


# In[4]:


get_ipython().system('pip install transformers')


# In[5]:


#import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from transformers.file_utils import is_tf_available, is_torch_available


# In[6]:





# In[7]:


PATH_DATA = os.path.join(Path.cwd(), 'data')
PATH_SUBM = os.path.join(Path.cwd(), 'submissions')


# In[7]:





# In[8]:


get_ipython().system('mkdir .kaggle')
get_ipython().system('touch .kaggle/kaggle.json')

api_token = {"username":"user","key":"api-key"}

with open('/content/.kaggle/kaggle.json', 'w') as file:
    json.dump(api_token, file)

get_ipython().system('chmod 600 /content/.kaggle/kaggle.json')
get_ipython().system('kaggle config path -p /content')


# In[9]:


get_ipython().system('mv .kaggle /root/')


# In[10]:


get_ipython().system('kaggle competitions download -c simplesentiment')


# Получаем верные метки. Для второй части нас здесь интересует df.target, seed и лямбда функция clean_text.

# In[11]:


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


# In[12]:


set_seed(16041961)


# In[13]:


df = pd.read_csv(os.path.join('./', 'products_sentiment_train.tsv'), 
                    header = None, 
                    index_col = None,
                    sep = '\t',
                   )
df.columns = ['text', 'target']
df.shape


# In[14]:


clean_text = lambda x: re.sub(r"\s+", ' ', 
                              re.sub(r"[\d+]", '',
                                     re.sub(r"[^\w\s]", '', x.lower()).strip()
                                    )
                             )


# Загружаем tokenizer.

# In[15]:


PRE_TRAINED_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


# In[16]:


max_length = 96


# In[17]:


tokenizer(['hello people twinky', 'world'], truncation=True, padding=True, max_length=max_length)


# In[17]:





# Загружаем модель.

# In[18]:


model = AutoModelForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME)


# In[ ]:





# Данные для прогноза.

# In[19]:


df_subm = pd.read_csv(os.path.join('./', 'products_sentiment_test.tsv'),
                        index_col = None,
                        sep = '\t',
                     )
df_subm.shape


# In[20]:


df_subm['text_cl'] = df_subm.text.map(clean_text)


# In[21]:


max_length = 96


# Подмассив задаем через idx = 0..4

# In[66]:


get_ipython().run_cell_magic('time', '', '#idx = 0..4\nidx = 4\nsubm_tokens = tokenizer(list(df_subm[\'text_cl\'].values[idx*100:(idx + 1)*100]), \n#subm_tokens = tokenizer(list(df_subm[\'text_cl\'].values), \n                        truncation=True, padding=True, max_length=max_length, \n                        return_tensors="pt"\n                        )')


# In[67]:


#subm_tokens


# Предсказываем для выбранного подмассива. Переводим в вероятности. Выбираем класс по максимому.

# In[68]:


get_ipython().run_cell_magic('time', '', 'outp = model(**subm_tokens)')


# In[69]:


#type(outp)


# In[70]:


#outp


# In[71]:


outp = outp[0].softmax(1)


# Сохраняем не класс, а все три вероятности. На случай если будем настраивать порог.

# In[72]:


with open(os.path.join('./', f'outp_p{idx}.pkl'), 'wb') as fd:
    pkl.dump(outp, fd)


# In[73]:


#outp


# Загружаем все подготовленные веростности.

# In[79]:


submit_y = list()
for idx in range(5):
    with open(os.path.join('./', f'outp_p{idx}.pkl'), 'rb') as fd:
        outp = pkl.load(fd)
  
     #submit_y += list(map(lambda x: x.argmax().item(), outp))
     submit_y += list(map(lambda x: 0 if x[0] > x[2] else 1, outp))


# In[75]:


len(submit_y)


# In[81]:


submit_y.count(0), submit_y.count(1),


# In[82]:


#outp_p1[0].softmax(1)[5].argmax().item()


# In[83]:


subm = pd.read_csv(os.path.join('./', 'products_sentiment_sample_submission.csv'))
subm.shape


# In[84]:


subm.y = submit_y


# In[85]:


subm.to_csv(os.path.join('./', 'twitter-roberta-base.csv'), index = False)


# In[1]:


#!ls


# In[ ]:





# In[ ]:





# In[86]:


from google.colab import files
files.download('twitter-roberta-base.csv') 


# In[ ]:


pb 0.89111

