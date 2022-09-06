#!/usr/bin/env python
# coding: utf-8

# In[1]:


## ноутбук выполняется на колабе


# In[2]:


import os
from typing import List
import json
from pathlib import Path
import re
import codecs
import random
import pickle as pkl

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# In[ ]:


os.environ['KAGGLE_USERNAME'] = 'user_name'
os.environ['KAGGLE_KEY'] = 'api_key'


# In[ ]:


get_ipython().system('kaggle competitions download -c morecomplicatedsentiment')


# По условиям задачи требуется в https://www.kaggle.com/c/morecomplicatedsentiment/ превзойти 0.85 по accuracy. Чтож.   
# Сейчас в nlp на голову всех выше трансформеры. Если поискать, то можно найти предобученный на рускоязычных отзывах   
# 'blanchefort/rubert-base-cased-sentiment-rurewiews'. В принципе просто сабмит на этой модели дает 0.87777, что уже   
# удовлетворяет условиям задачи (точнее не совсем просто сабмит. модель подразумеваем еще и нейтральную тональность.   
# так что просто будем считать: если вероятность позитивного больше, чем негативного - отзыв позитивный. В противном   
# случае - негативный. Игнорируем нейтральный.)

# Но это выглядит странным. По заданию все же подразумевается обучение моделей. Хорошо, дообучим указанную модель на наших данных.   
# Посмотрим, что мы получим.

# In[ ]:


get_ipython().system('pip install transformers')


# In[ ]:


#!pip install datasets


# In[ ]:


import torch
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments

#from datasets import load_metric


# Подключаем гугл диск с данными

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


PATH_DATA = os.path.join(Path.cwd(), 'drive', 'MyDrive')
PATH_SUBM = os.path.join(Path.cwd(), 'submissions')


# In[ ]:


#ls /content/drive/MyDrive


# In[ ]:





# ## Загружаем предобрабатываем собранные нами данные

# In[ ]:


df = pd.read_csv(os.path.join(PATH_DATA,'ru_train.csv'))
df.shape


# In[ ]:


df.head()


# Очищаем данные

# In[ ]:


#clean_text = lambda x:' '.join(re.sub('\n|\r|\t|[^а-яa-z]', ' ', x.lower()).split())
clean_text = lambda x:' '.join(re.sub('\n|\r|\t|[^а-я]', ' ', x.lower()).split())


# In[ ]:


df['clean_text'] = df.review.map(clean_text)


# Подразумевается, что 2 класса: позитивный (возьмем оценки 4 и 5) и негативный (возьмем оценки 1, 2 и 3). 

# In[ ]:


# 0: NEUTRAL
# 1: POSITIVE
# 2: NEGATIVE


# In[ ]:


#set_tone = lambda x: 'pos' if int(x) >= 4 else 'neg'
set_tone = lambda x: 1 if int(x) >= 4 else 2
df['tone'] = df.rating.map(set_tone)


# Перемешиваем отзывы

# In[ ]:


df = df.sample(frac=1).reset_index(drop=True)
df.drop(['Unnamed: 0'], axis = 1, inplace = True)
df.head()


# In[ ]:





# In[ ]:





# Прдготовливаем тренировочную и тестовую выборки

# In[ ]:


train, test, train_lbl, test_lbl = train_test_split(df.clean_text, df.tone, 
                                                   test_size = 0.2, stratify = df.tone,
                                                  random_state = 354274)
train.shape, test.shape, train_lbl.shape, test_lbl.shape


# In[ ]:


train_lbl.value_counts(), test_lbl.value_counts()


# Так же на интререс попробуем еще несколько моделей, обученных на не отзывах. Но просто модели, без дообучения.

# In[ ]:


#task='sentiment'
#PRE_TRAINED_MODEL_NAME = f"blanchefort/rubert-base-cased-{task}"

#PRE_TRAINED_MODEL_NAME = 'blanchefort/rubert-base-cased-sentiment-rusentiment'
PRE_TRAINED_MODEL_NAME = 'blanchefort/rubert-base-cased-sentiment-rurewiews'

MODEL_FOLDER = 'ru-blanchefort-rurewiews2'

MAX_LENGTH = 512


# Загружаем токенайзер, переводящий данные в embeddings, модель.   
# Переводим наши данные в torch.tensor, необходимый для обучения.

# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


# In[ ]:


train_tokens = tokenizer(list(train.values), truncation=True, padding=True, max_length=MAX_LENGTH)
test_tokens  = tokenizer(list(test.values),  truncation=True, padding=True, max_length=MAX_LENGTH)


# In[ ]:


class TonalityDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels: List[str]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> int:
        """
        Получение метки класса из тензора
        args:
            idx - индекс требуемой метки класса
        return:
            ште - метка класса
        """
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        
        return item

    def __len__(self) -> int:
        return len(self.labels)


# In[ ]:


train_dataset = TonalityDataset(train_tokens, train_lbl.values)
test_dataset  = TonalityDataset(test_tokens,  test_lbl.values)


# In[ ]:


# 0: NEUTRAL
# 1: POSITIVE
# 2: NEGATIVE
def compute_metrics(pred) -> Dict:
    """
    Расчет метрики roc-auc для расчитанных и истиных значениях классов
    """
    
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate roc_auc_score using sklearn's function
    rocauc = roc_auc_score(labels, preds)
    
    return {
        'roc-auc': rocauc,
    }


# In[ ]:


training_args = TrainingArguments(
    output_dir=f'/content/drive/MyDrive/models/{MODEL_FOLDER}/results',          # output directory
    num_train_epochs=1,              # total number of training epochs
    #per_device_train_batch_size=16,  # batch size per device during training
    #per_device_eval_batch_size=20,   # batch size for evaluation
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=300,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    #logging_dir=f'/content/drive/MyDrive/models/{MODEL_FOLDER}/logs',            # directory for storing logs
    #logging_first_step  = True, 
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    #metric_for_best_model = compute_metrics,
    greater_is_better = True,    # if use roc-auc inside 'metric_for_best_model'
    logging_steps=50,               # log & save weights each logging_steps
    evaluation_strategy="steps",    #"epoch" # evaluate each `logging_steps`

    seed = 112,
)


# In[ ]:


#no roc-auc metric on current moment
#load_metric('rocauc')
#https://github.com/huggingface/datasets/tree/master/metrics


# In[ ]:


model = AutoModelForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME,)
model.classifier


# Будет переобучен слой классификатора.   
# Выходной слой подразумевает выход на 3 класса. Изменить возможно, но легко не удалось, необходимо лезть в дебри pytorch, но я его пока не знаю.   
# Ничего страшного. У нас нет нейтральных отзывов. Так что вероятность нейтральных отзывов должна занулиться.   

# In[ ]:


#не использовулось в финальном решении, т.к. веса джля не исполльзуемого класса и так будут занулены

#заменяем голову модели на выход для 2х классов
#model.classifier = torch.nn.Linear(in_features=768, out_features=2, bias=True)
#model.classifier


# In[ ]:


trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,           # evaluation dataset
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
)


# Считаем на cpu. В зависимости от нагрузки colab может успеть просчитать все (благо поставил только 1 эпоху).
# У меня не удавалось, так что сохраняем промежуточные состояния на google drive и, после сброса colab 8ми часовой сессии,   
# вновь запускаем подсчет.

# In[ ]:


#trainer.train()


# In[ ]:


trainer.train(resume_from_checkpoint = True)


# In[ ]:


model.save_pretrained(os.path.join('/content/drive/MyDrive/models/', f'{submname}'))


# In[ ]:





# Делаем сабмит.

# In[ ]:


with codecs.open(os.path.join('.', 'test.csv'), encoding='utf-8') as fd:
    data = fd.read()


# In[ ]:


data = data.lower().split('</review>\n\n<review>')
data[0] = data[0][8:]
data[-1] = data[-1][:-11]

len(data)


# In[ ]:


data = list(map(clean_text, data))


# In[ ]:


#data = list(map(lambda x: ' '.join([el for el in x.split() if el not in russian_stopwords]), data))


# In[ ]:





# ### Посмотрим на получившиеся данные

# In[ ]:


data[random.randrange(100)]


# In[ ]:


np.mean([len(el.split()) for el in data])


# 

# In[ ]:


subm = pd.read_csv(os.path.join('.', 'sample_submission.csv'))


# In[ ]:


subm.y.value_counts()


# In[ ]:


#subm.y = ['pos'] * subm.shape[0]


# In[ ]:


@torch.no_grad()
def predict(text: List[str]) -> List[float, float]:
    """
    Предсказываем метки класса для отзывов
    args:
        text - тексты отзывов
    return:
        вероятности классов
    """
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
    #predicted = torch.argmax(predicted, dim=1).numpy()
    return predicted


# In[ ]:


get_ipython().run_cell_magic('time', '', 'pred = predict(data)')


# In[ ]:


# по вероятностям классов разделим на позитивные и неготивные отзывы
for idx in subm.index:
    if pred[idx][1].item() > pred[idx][2].item():
        subm.loc[idx, 'y'] = 'pos'
    else:
        subm.loc[idx, 'y'] = 'neg'

subm.y.value_counts()


# In[ ]:





# Сохраним в сабмит

# In[ ]:


submname = 'ru-blanchefort-rurewiew_tuned_1epochs'
subm.to_csv(os.path.join('.', f'{submname}.csv'), index = False)


# In[ ]:


from google.colab import files
files.download(f'{submname}.csv') 


# In[ ]:





# ## Получили 0.97777 на pb

# In[ ]:





# In[ ]:


results:
XXXXXXX ru-blanchefort-rurewiew_stem
neg    87
pos    13

0.72222 ru-blanchefort-sentiment 
neg    26
pos    74

0.78888 ru-blanchefort-rusentiment 
neg    70
pos    30

0.81111 ru-blanchefort-rurewiew_lemm
neg    68
pos    32

0.85555 ru-blanchefort-rurewiew_nosw
neg    65
pos    35

0.87777 ru-blanchefort-rurewiew
neg    61
pos    39

0.97777 ru-blanchefort-rurewiew_tuned_1epochs
neg    49
pos    51

