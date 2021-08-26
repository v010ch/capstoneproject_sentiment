#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np

from statistics import mean, stdev

from tqdm import tqdm
tqdm.pandas()

from sklearn.pipeline import Pipeline


# In[1]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


# In[60]:


import nltk
#nltk.download('movie_reviews')
#nltk.download('stopwords')

from nltk.corpus import movie_reviews


# In[4]:


PATH_TO_DATA = '/home/voloch/nltk_data/corpora/movie_reviews'
ANSW = './answers'


# In[5]:


neg_ids = movie_reviews.fileids('neg')
pos_ids = movie_reviews.fileids('pos')


# In[6]:


neg_feats = [movie_reviews.words(fileids=[f]) for f in neg_ids]


# In[7]:


neg_feats[0]


# In[8]:


for f in neg_ids[:3]:
    print(f)


# In[9]:


len(neg_ids), len(pos_ids)


# In[ ]:





# In[10]:


all_reviews = []
all_ratings = []


# In[11]:



for elmnt in tqdm(neg_ids):
    with open(os.path.join(PATH_TO_DATA, elmnt)) as fd:
        tmp_data = fd.read()
    all_reviews.append(tmp_data)
    all_ratings.append(0)
    
for elmnt in tqdm(pos_ids):
    with open(os.path.join(PATH_TO_DATA, elmnt)) as fd:
        tmp_data = fd.read()
    all_reviews.append(tmp_data)
    all_ratings.append(1)


# In[12]:


data = pd.DataFrame({'review': all_reviews, 'rating': all_ratings})


# # WEEK 1

# In[13]:


def write_answer_to_file(answer, file_address):
    if isinstance(answer, list) or isinstance(answer, np.ndarray):
        with open(os.path.join(ANSW, file_address), 'w') as out_f:
            for idx, elmnt in enumerate(answer):
                if idx == 0:
                    out_f.write(str(elmnt))
                else:
                    out_f.write(' ' + str(elmnt))
    else:
        with open(os.path.join(ANSW, file_address), 'w') as out_f:
            out_f.write(str(answer))


# In[14]:


answ1 = data.shape[0]
write_answer_to_file(answ1, 'answer1_1.txt')
answ1


# In[15]:


answ2 = data[data.rating == 1].shape[0] / data.shape[0]
write_answer_to_file(answ2, 'answer1_2.txt')
answ2


# In[16]:


CntVect = CountVectorizer()


# In[17]:


CntVect.fit(data.review.values)


# In[18]:


answ3 = len(CntVect.get_feature_names())
write_answer_to_file(answ3, 'answer1_3.txt')
answ3


# In[19]:


X_train = CntVect.transform(data.review.values)


# In[20]:


y_train = data.rating.values


# In[21]:


model = LogisticRegression()


# In[22]:


answ4 = cross_val_score(model, X_train, y_train, scoring = 'accuracy')
answ4 = sum(answ4) / len(answ4)
write_answer_to_file(answ4, 'answer1_4.txt')
answ4


# In[ ]:





# In[ ]:





# In[23]:


model = LogisticRegression()


# In[24]:


answ5 = cross_val_score(model, X_train, y_train, scoring = 'roc_auc')
answ5 = sum(answ5) / len(answ5)
write_answer_to_file(answ5, 'answer1_5.txt')
answ5


# In[25]:


clf = LogisticRegression()


# In[26]:


clf.fit(X_train, y_train)


# In[27]:


len(CntVect.vocabulary_)


# In[28]:


len(CntVect.get_feature_names())


# In[29]:


answ_list = [(el0, el1) for el0, el1 in zip(CntVect.get_feature_names(),clf.coef_[0])]


# In[30]:


answ_list.sort(key = lambda x: abs(x[1]), reverse = True)


# In[31]:


answ_list[:10]


# In[32]:


answ6 = []
for el, _ in answ_list[:2]:
    answ6.append(el)

write_answer_to_file(answ6, 'answer1_6.txt')
answ6


# In[ ]:





# In[ ]:





# # WEEK 2

# In[33]:


stepsCntVect = [('vect', CountVectorizer()), ('lr', LogisticRegression())]
pipelCntVect = Pipeline(stepsCntVect)


# In[34]:


stepsTfIdfVect = [('vecttf', TfidfVectorizer()), ('lr', LogisticRegression())]
pipelTfIdfVect = Pipeline(stepsTfIdfVect)


# In[35]:


#answ2_1_1 = cross_val_score(pipelCntVect, data.review.values, y_train, scoring = 'accuracy', cv = 5)
answ2_1_1 = cross_val_score(pipelCntVect, data.review.values, y_train, cv = 5)
answ2_1_1


# In[36]:


#answ2_1_2 = cross_val_score(pipelTfIdfVect, data.review.values, y_train, scoring = 'accuracy', cv = 5)
answ2_1_2 = cross_val_score(pipelTfIdfVect, data.review.values, y_train, cv = 5)
answ2_1_2


# In[37]:


answ2_1 = [mean(answ2_1_1), stdev(answ2_1_1), mean(answ2_1_2), stdev(answ2_1_2)]
write_answer_to_file(answ2_1, 'answer2_1.txt')
answ2_1


# In[39]:


paramCntVect = {
    'vect__min_df': [10, 50]
}


# In[41]:


search = GridSearchCV(pipelCntVect, param_grid = paramCntVect, cv = 5)
answ2_2 = search.fit(data.review.values, y_train)
#answ2_2


# In[ ]:


#answ2_2 = cross_val_score(pipelCntVect, data.review.values, y_train, fit_params = paramCntVect, cv = 5)
#answ2_2


# In[42]:


answ2_2.cv_results_


# In[43]:


answ2_2.cv_results_['mean_test_score']
write_answer_to_file(answ2_2.cv_results_['mean_test_score'], 'answer2_2.txt')
answ2_2.cv_results_['mean_test_score']


# In[ ]:





# In[68]:


CntVect = CountVectorizer()
CntVect.fit(data.review.values)
X_train = CntVect.transform(data.review.values)


# In[69]:


clfLR = LogisticRegression()
resLR = cross_val_score(clfLR, X_train, y_train, cv = 5)
resLR


# In[70]:


clfSVC = LinearSVC()
resSVC = cross_val_score(clfSVC, X_train, y_train, cv = 5)
resSVC


# In[71]:


clfSGD = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, 
                       fit_intercept=True, max_iter=5, shuffle=True, verbose=0, 
                       epsilon=0.1, n_jobs=1, random_state=42, learning_rate='optimal', 
                       eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, average=False)
resSGD = cross_val_score(clfSGD, X_train, y_train, cv = 5)
resSGD


# In[72]:


answ2_3 = min([mean(resLR), mean(resSVC), mean(resSGD)])
write_answer_to_file(answ2_3, 'answer2_3.txt')
answ2_3


# In[ ]:





# In[62]:


len(nltk.corpus.stopwords.words('english'))


# In[73]:


CntVect = CountVectorizer(stop_words=nltk.corpus.stopwords.words('english'))
CntVect.fit(data.review.values)
X_train_nltk = CntVect.transform(data.review.values)


# In[74]:


clfstopwLR = LogisticRegression()
retnltkstopw = cross_val_score(clfstopwLR, X_train_nltk, y_train, cv = 5)
retnltkstopw


# In[76]:


CntVect = CountVectorizer(stop_words='english')
CntVect.fit(data.review.values)
X_train_sklearn = CntVect.transform(data.review.values)


# In[77]:


clfstopw2LR = LogisticRegression()
retnltkstopw2 = cross_val_score(clfstopw2LR, X_train_sklearn, y_train, cv = 5)
retnltkstopw2


# In[79]:


answ2_4 = [mean(retnltkstopw), mean(retnltkstopw2)]
write_answer_to_file(answ2_4, 'answer2_4.txt')
answ2_4


# In[ ]:





# In[80]:


CntVect = CountVectorizer(ngram_range=(1, 2))
CntVect.fit(data.review.values)
X_train_wide_words = CntVect.transform(data.review.values)


# In[82]:


clfLR_wide_word = LogisticRegression()
ret_wide_word = cross_val_score(clfLR_wide_word, X_train_wide_words, y_train, cv = 5)
ret_wide_word


# In[83]:


CntVect = CountVectorizer(ngram_range=(3, 5), analyzer='char_wb')
CntVect.fit(data.review.values)
X_train_wide_chars = CntVect.transform(data.review.values)


# In[84]:


clfLR_wide_char = LogisticRegression()
ret_wide_char = cross_val_score(clfLR_wide_char, X_train_wide_chars, y_train, cv = 5)
ret_wide_char


# In[85]:


answ2_5 = [mean(ret_wide_word), mean(ret_wide_char)]
write_answer_to_file(answ2_5, 'answer2_5.txt')
answ2_5


# In[ ]:





# In[ ]:





# In[ ]:





# for myself

# In[88]:


CntVect = CountVectorizer(ngram_range=(1, 2), stop_words=nltk.corpus.stopwords.words('english'))
CntVect.fit(data.review.values)
X_train_wide_words_n_stop = CntVect.transform(data.review.values)


clfLR_wide_word_n_stop = LogisticRegression()
ret_wide_word_n_stop = cross_val_score(clfLR_wide_word_n_stop, X_train_wide_words_n_stop, y_train, cv = 5)
ret_wide_word_n_stop


# In[89]:


mean(ret_wide_word_n_stop)


# In[ ]:





# # WEEK 3

# In[ ]:




