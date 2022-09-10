#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from pathlib import Path
import pickle as pkl
import time
from tqdm import notebook  as pbar
from typing import Optional

from selenium import webdriver
#from selenium.webdriver.common.keys import Keys
#from selenium.webdriver.common.by import By


# In[ ]:


from selenium.webdriver.common.by import By


# In[ ]:


from pandas import DataFrame, read_csv, concat
import numpy as np
from scipy import stats as sts


# In[ ]:


#import seaborn as sns


# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')


# ### Сбор отзывов для построения своего датасета

# с форума проекта: "Еще можно парсить https://irecommend.ru/ но будьте осторожны, он не такой терпеливый, если ставить задержку меньше 10 сек., то забанит.    
# Да, и в нем надо парсить html. Будьте готовы ждать пару-тройку десятков часов =)"   
# Это вызов)))) Попробую получать быстрее.

# Из одной презентации на тему "как получить данные, когда вам их не хотят отдавать" я помню, что основная причина блока - это не соответствие   
# заголовка заголовку браузера. И в requests и в spacy есть возможность выставлять заголовки, но это приходится делать вручную, да и при смене   
# в браузерах их придется обновлять. Так же в своей практике сталкивался с тем, что часть страницы может быть сгенерирована js скриптом, это   
# приводит к тому, что requests и bs4 эти данные не увидят. Для обхода этого используют selenium, да вот руки не доходили попробовать.   
# Теперь есть шанс - будем использовать selenium.

# Хорошо, с заголовками разобрались, вторая известная мне причина блока - запросы чаще определенного промежутка. В презентации говорилось о 3х сек.   
# Достаточно долго, если нам необходимо собрать много отзывов. Попробуем ускорить.   
# Будем генерировать запросы с частотой из 3х распределений. В целом, нам нужны распределения скошенные влево. Используем gamma, chi2 и нормальное (куда уж без него).   
# Но так мы будем получать и значения задержки около 0, так и в 20 сек и более. Так что установим ограничение: запросы не могут быть меньше MIN_DELAY (мы же не   
# хотим попасть в блок и ждать сутки или более пока нас разблокируют, мы хотим побыстрее собрать данные и закончить неделю) и дольше MAX_DELAY.

# Все, с исходными условиями разобрались - пробуем различные варианты.

# In[ ]:


PATH_DATA = os.path.join(Path.cwd(), 'data')


# In[ ]:


MIN_DELAY = 1.71#2.17 #2.673
MAX_DELAY = 4.8 #7.22 #9.181


# In[ ]:


class UserEmulate:
    def __init__(self, inp_min_delay: float, inp_max_delay: float) -> None:
        self.min_delay = inp_min_delay
        self.max_delay = inp_max_delay
        
        self.last_time = time.time()
        self.numb_load = 0


        
        
    def reset(self, inp_min_delay: Optional[float], inp_max_delay: Optional[float]) -> None:
        """
        Сброс парметров и выставление новых мин и макс задержки
        args
            inp_min_delay - минимальная задержка между загрузками страниц (опционально)
            inp_max_delay - максимальная задержка между загрузками страниц (опционально)
        """
        self.last_time = time.time()
        self.numb_load = 0
        
        if isinstance(inp_min_delay, float):
            self.min_delay = inp_min_delay
            
        if isinstance(inp_max_delay, float):
            self.max_delay = inp_max_delay
       
    
    
    def updatecurrentstate(self):
        self.last_time = time.time()
        self.numb_load += 1
        
        
    def pauserealuseremulate(self) -> None:
        """
        Эмуляция задержки между кликами пользователя.
        Каждый седьмой клик из нормального распределения
        Каждый третий (при не кратности 7) из хи-квадрат
        Остальные из гамма
        """
        if self.numb_load %7 == 0:
            pause_time = sts.norm.rvs(loc=2, scale=3, size=1)[0]
        elif self.numb_load %3 == 0:
            pause_time = sts.chi2.rvs(df = 1.7, loc = 0, scale = 1, size=1)[0]
        else:
            pause_time = sts.gamma.rvs(a = 1, loc = 1, scale = 2, size=1)[0]

        if (time.time() - self.last_time) > pause_time:
            self.updatecurrentstate()
            return

        if pause_time >= self.min_delay and pause_time <= self.max_delay:
            #print(pause_time)
            time.sleep(pause_time - abs(time.time() - self.last_time))
            self.updatecurrentstate()
            pass
        else:
            pauserealuseremulate()

        return


# In[ ]:





# Загрузка списка уже загруженных отзывов, если загрузка осуществляется в несколько запусков

# In[ ]:


if os.path.exists(os.path.join(PATH_DATA, 'loaded_links.pkl')):
    with open(os.path.join(PATH_DATA, 'loaded_links.pkl'), 'rb') as fd:
        loaded_links = pkl.load(fd)    
else:
    print('Create new dict!')
    loaded_links = dict({'pages':   set(), 
                         'phones':  set(), 
                         'reviews': set(),
                        })


# использовтаь будем Firefox. т.к. мы хотим мимикрировтаь под пользователя.   
# а при проверке на https://www.whatismybrowser.com/detect/what-http-headers-is-my-browser-sending?sort=dont-sort   
# у Chrome отображается различный заголовок у браузера которым пользуюсь и у selenium
# 
# идти будем не по отзывам, а по моделям телефона. в обоих случаях отображается 100 страниц - всего 2000 отзывов/моделей,   
# но в каждой модели можно посмотреть все отзывы о ней - отзывов будет многим больше.

# In[ ]:


#headers check
#SITE = 'https://www.whatismybrowser.com/detect/what-http-headers-is-my-browser-sending?sort=dont-sort'
#SITE = "https://www.supermonitoring.com/blog/check-browser-http-headers/"

#get reviews
#SITE = 'https://irecommend.ru/catalog/reviews/55'
#SITE = 'https://irecommend.ru/catalog/list/55'

sites = ['https://irecommend.ru/catalog/list/55'] + [f'https://irecommend.ru/catalog/list/55?page={ind}' for ind in range(1, 100)]


# In[ ]:


driver = webdriver.Firefox(executable_path = "C:\\WebDrivers\\bin\\geckodriver")
ue = UserEmulate(MIN_DELAY, MAX_DELAY)


# In[ ]:





# Загружаем отзывы

# In[ ]:


reviews_page_df = DataFrame()

#for page_number, url in enumerate(sites[:9]):
for page_number, url in pbar.tqdm(enumerate(sites[:40]), position=0, leave = True):
    if url in loaded_links['pages']:
        print(f'page link {url} already scraped!')
        continue
        
    print(url)
    # pause for emulate user behavior
    ue.pauserealuseremulate()
    # load url
    driver.get(url)
    # reviews by phones
    #phone_names = driver.find_elements_by_class_name('title')
    phone_names = driver.find_elements(By.CLASS_NAME, 'title')
    phone_names = [el.text for el in phone_names]
    # some data doubled. stay only one from two
    #phones = driver.find_elements_by_class_name('read-all-reviews-link')
    phones = driver.find_elements(By.CLASS_NAME, 'read-all-reviews-link')
    phones = [el.get_property('href') for idx, el in enumerate(phones) if idx%2 == 0]
    # number of reviews on this url
    #ttl_size = driver.find_elements_by_class_name('counter')
    ttl_size = driver.find_elements(By.CLASS_NAME, 'counter')
    ttl_size = [int(el.get_property('innerHTML')) for idx, el in enumerate(ttl_size) if idx%2 == 0]
    
    tmp = [(el0, el1) for idx, (el0, el1) in enumerate(zip(phones, ttl_size))]
    # there need to check if the review has already scraped
    # and drop already scrapped
    phones   = [el0 for el0, el1 in tmp]
    ttl_size = [el1 for el0, el1 in tmp]
    
    reviews_page_df = DataFrame(index = range(sum(ttl_size)), columns = ['phone', 'review', 'rating', 'link'])
    
    index = 0
    #for idx in tqdm(range(len(phones))):
    for idx in pbar.tqdm(range(len(phones)), position=1, leave = True):
    #for idx in pbar.tqdm(range(3), position=0, leave = True):
        #print(phone_names[idx], phones[idx])
        ue.pauserealuseremulate()
        driver.get(phones[idx])
        
        #reviews = driver.find_elements_by_class_name('more')
        reviews = driver.find_elements(By.CLASS_NAME, 'more')
        reviews = [el.get_property('href') for el in reviews]
        
        #for review_link in pbar.tqdm(reviews, position=1, leave = True):
        for review_link in reviews:
            ue.pauserealuseremulate() 
            driver.get(review_link)
            
            # zero rating - just ad
            #rating = driver.find_elements_by_class_name("fivestarWidgetStatic")[1]
            rating = driver.find_elements(By.CLASS_NAME, "fivestarWidgetStatic")[1]
            #rating = len(rating.find_elements_by_class_name('on'))
            rating = len(rating.find_elements(By.CLASS_NAME, 'on'))
            #print(rating)
            #text = driver.find_elements_by_class_name("views-field-teaser")[0].text
            text = driver.find_elements(By.CLASS_NAME, "views-field-teaser")[0].text
            #print(text[:20])
            reviews_page_df.loc[index, 'phone']  = phone_names[idx]
            reviews_page_df.loc[index, 'review'] = text
            reviews_page_df.loc[index, 'rating'] = rating
            reviews_page_df.loc[index, 'link']   = review_link
            index += 1
            
    
    # save after each page
    reviews_page_df.to_csv(os.path.join(PATH_DATA, f'reviews_own_page{page_number}.csv'))
    loaded_links['pages'].add(url)
    with open(os.path.join(PATH_DATA, 'loaded_links.pkl'), 'wb') as fd:
        pkl.dump(loaded_links, fd)


# Итого получилось порядка 2с на отзыв. Т.е. за 24 часа более 40000. Хорошо. Так же мы обошли ограничение в 10 сек и даже в 3 сек.

# In[ ]:





# Сохранял каждую страницу в отдельный файл. Совместим их.

# In[ ]:


review_df = DataFrame(columns = ['phone', 'review', 'rating', 'link'])
#review_df


# In[ ]:


for file in os.listdir('data'):
    if file.startswith('review'):
        print(file)
        tmp_df = read_csv(os.path.join(PATH_DATA, file), index_col = 0)
        review_df = concat((review_df, tmp_df), 
                           ignore_index = True
                          )
print(review_df.shape)
review_df.dropna(axis = 0, inplace = True)
print(review_df.shape)


# In[ ]:


review_df.sample(10)


# Что нам известно об данном нам датасете:   
# - отзывы о телефонах;   
# - есть много жаргона и названий компаний на русском, много слитых слов, опечатки:
#   'кверти', 'рут', 'очень травится', 'полная френь', 'ценанадежность'
# - на первый взгляд очень мало положительных отзывов;
# - минимальная длина отзыва - 16, средняя - 116, максимальная - 531;  
# - все 'neg' дают 0.48888. все 'pos' - 0.51111.

# In[ ]:





# Необходимо собрать датасет примерно похожий на предоставленный:   
# длина не более 531, пропорции pos/neg ~50/50.   
# Отбросим слишком длинные отзывы из нашей выборки, затем уберем часть позитивных отзывов,   
# для уравнивания пропорций с оставшимися негативными после удаления длинных отзывов.   
# У нас в выборке 5 звезд, т.к. класса в тестовой выборке 2, без нейтрального, будем считать   
# все 5 и 4 звезд - за позитивные отзывы, 3 и менее - за негативные.

# In[ ]:


review_df['review_length'] = review_df.review.map(lambda x: len(x.split(' ')))


# In[ ]:


hist(review_df.review_length, bins = 20)


# In[ ]:


review_df.rating.value_counts()


# In[ ]:


review_df.phone.value_counts()


# In[ ]:


#lens = [len(review_df.loc[idx, 'review'].split(' ')) for idx in review_df.dropna(axis = 0).index]


# 

# Максиммальную длину отзыва возьмем за 600. т.к. если взять 500 - то теряются почти 300 негативных отзывов.

# In[ ]:


train = review_df.query('review_length < 600')
train.reset_index(inplace = True)
train = train.drop(['index'], axis = 1)
train.shape


# In[ ]:


train.query('rating <= 3').shape, train.query('rating > 3').shape


# удалим 10000 случайных положительных отзывов.   
# при получении не очень хорошего итогового результата можно повторять данный этап несколько раз для получения разных вборок.

# In[ ]:


index_to_drop = np.random.choice(train.query('rating > 3').index, 10000, replace = False)
len(set(index_to_drop))


# In[ ]:


train.drop(train.index[index_to_drop], axis = 0, inplace = True)#.rating.value_counts()
train.shape


# In[ ]:


train.rating.value_counts()


# In[ ]:


train.to_csv(os.path.join(PATH_DATA, 'ru_train.csv'))


# Проверим только одну гипотезу: корреляция между длинной отзыва и оценкой

# In[ ]:


train[['rating', 'review_length']].corr()


# Нет, корреляция не наблюдается.   
# Все, отзывы собраны, перейдем к обучению модели.

# In[ ]:




