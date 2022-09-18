import os
import pickle as pkl
import re
from typing import Dict

try:
    import nltk.stem as st
    #st.WordNetLemmatizer()('done')
    #st.RSLPStemmer()('done')
except:
    pass

try:
    from transformers import pipeline
except:
    pass


class ModelClassTemplate():
    """
    Шсблон для класса определения тональности
    """
    def __init__(self, state: Dict, model_name: str) -> None:
        with open(model_path, 'rb') as fd:
            self.model = pkl.load(df)
            self.state = state



    def clean_text(self, inp_text: str) -> str:
        """
        Очистка текста
        args:
            inp_test - отзыв
        return:
            str - отзыв, очищенный от всего кроме слов
        """
        #cl_txt = lambda x: re.sub(r"\s+", ' ', 
        #                          re.sub(r"[\d+]", '',
        #                                re.sub(r"[^\w\s]", '', x.lower()).strip()
        #                                )
        #                     )
        return re.sub(r"\s+", ' ', 
                      re.sub(r"[\d+]", '',
                             re.sub(r"[^\w\s]", '', inp_text.lower()).strip()
                             )
                     )



    def prepare_text(self, inp_text: str) -> str:
        """
        Очистка текста
        args:
            inp_test - отзыв
        return:
            str - отзыв, очищенный от всего кроме слов
        """
        return self.clean_text(inp_text)
        
        
        
    def predict_tonality(self, inp_text: str):
        """
        Шасблон для функции определения тональности
        """
        text = self.prepare_text(inp_text)
        pred = self.model.predict(text)
        
        return pred
    


    
class ModelTfIdf(ModelClassTemplate):
    """
    Класс определения тональности через tf-idf
    """
    def __init__(self, state: Dict, model_name: str) -> None:
        with open(os.path.join('./', 'models', model_name+'_model.pkl'), 'rb') as fd:
            self.model = pkl.load(fd)
        with open(os.path.join('./', 'models', model_name+'_token.pkl'), 'rb') as fd:
            self.tokenizer = pkl.load(fd)
        self.state = state
        
        if self.state['stem']:
            self.lemm = st.WordNetLemmatizer()
            self.stem = st.RSLPStemmer()


            
    def prepare_text(self, inp_text: str) -> str:
        """
        Очистка текста
        args:
            inp_test - отзыв
        return:
            str - отзыв, очищенный от всего кроме слов/
                  в случапе признака stem = True нгастройках еще и лумматизированный и стем
        """
        if self.state['stem']:
            preprocessed = self.clean_text(inp_text)
            preprocessed = ' '.join([self.lemm.lemmatize(el) for el in preprocessed.split()])
            preprocessed = ' '.join([self.stem.stem(el) for el in preprocessed.split()])
            return preprocessed
        else:
            return self.clean_text(inp_text)
        
        
        
    def predict_tonality(self, inp_text: str) -> str:
        """
        Получение тональности отзыва
        args:
            inp_text - входящий очищенный текст отзыва
        return:
            str - тональность отзыва positive / negative / neutraд
        """
        preprocessed_text = self.prepare_text(inp_text)
        text_embeddings = self.tokenizer.transform([preprocessed_text])
        pred = self.model.predict_proba(text_embeddings)
        #print(pred)
        #print(abs(pred[0][0] - pred[0][1]))
        if abs(pred[0][0] - pred[0][1]) < 0.1:
            return 'neutral'
        elif pred[0][0] > pred[0][1]: 
            return 'negative'
        else:
            return 'positive'
        
        


class ModelROBERTA(ModelClassTemplate):
    """
    Класс для определения тональности для обеих версий (как тяжеловестной так и легковестной) ROBERT
    """
    def __init__(self, state: Dict, model_name: str):
        #super().__init__(model_path)
        self.model = pipeline("sentiment-analysis", model = model_name)
        self.state = state

        
        
    def predict_tonality(self, inp_text: str) -> str:
        """
        Получение тональности отзыва
        args:
            inp_text - входящий очищенный текст отзыва
        return:
            str - тональность отзыва positive / negative / neutral
        """
        pred = self.model(inp_text)
        #print(pred)
        if self.state['roberta'] == 'base':
            if pred[0]['label'] == 'LABEL_0':
                return 'negative'
            elif pred[0]['label'] == 'LABEL_1':
                return 'neutral'
            else:
                return 'positive'
        else:
            if pred[0]['label'] == 'NEGATIVE':
                return 'negative'
            else:
                return 'positive'
