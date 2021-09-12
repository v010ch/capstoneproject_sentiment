#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import requests
from bs4 import BeautifulSoup as soup


# In[ ]:





# In[2]:


task_a = 'https://en.wikipedia.org/wiki/Bias-variance_tradeoff'
task_b = 'https://en.wikipedia.org/wiki/Category:Machine_learning_algorithms'


# In[ ]:





# ### из статьи https://en.wikipedia.org/wiki/Bias-variance_tradeoff все заголовки верхнего уровня

# К заголовку первого уровня по идее относится только h1. Как то слишком упрощено получается.   
# Пройдемся по всем от h1 до h6. Просто на интерес.

# При этом часть заголовков общая для почти всех страниц на вики. Исключим их в процессе.

# In[3]:


stop_headers = set(['Personal tools', 'Namespaces', 'Variants', 'Views', 'More', 'Search', 'Navigation',
                   'Contribute', 'Tools', 'Print/export', 'Languages', 'Navigation menu', 'Contents',
                    'See also[edit]', 'References[edit]',
                   ])


# In[4]:


print(task_a)
req = requests.get(task_a)
#page = soup(req.text, 'lxml')
page = soup(req.text, 'html')


# In[5]:


for rank in range(6):
    print(f'h{rank+1}')
    finded = page.findAll(f'h{rank+1}')
    
    if len(finded) != 0:
        for idx in range(len(finded)):
            if finded[idx].text.strip() not in stop_headers:
                if finded[idx].text.endswith('[edit]'):
                    print(finded[idx].text[:-6])
                else:
                    print(finded[idx].text)


# Мы получили заголовки разделов, относящиеся только к теме статьи.

# In[ ]:





# In[ ]:





# ### со страницы https://en.wikipedia.org/wiki/Category:Machine_learning_algorithms названия всех статей в категории Machine Learning Algorithms

# In[6]:


print(task_b)
req = requests.get(task_b)
#page = soup(req.text, 'lxml')
page = soup(req.text, 'html')


# In[7]:


finded = page.findAll('div', attrs={'class':'mw-category-group'})


# In[8]:


len(finded)


# In[9]:


total = 0
for idx in range(len(finded)):
    articles = finded[idx].findAll('a')
    for el in range(len(articles)):
        print(articles[el].text)
        total += 1


# In[10]:


print(total)


# Посмотрим, можно ли было найти названия статей проще.

# In[11]:


finded = page.findAll('a')


# In[12]:


len(finded)


# Очевидно проще не выходит, т.к. собирает все ссылки на страницы, а не только необходимых разделов.

# In[ ]:




