import os
import pickle as pkl
import re

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
    def __init__(self, state, model_name):
        with open(model_path, 'rb') as fd:
            self.model = pkl.load(df)
            self.state = state
            
    def clean_text(self, inp_text):
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
        #return cl_txt(inp_text)
    
    def prepare_text(self, inp_text):
        return clean_text(inp_text)
        
    def predict_tonality(self, inp_text):
        text = prepare_text(inp_text)
        pred = self.model.predict(text)
        return pred
    


class ModelTfIdf(ModelClassTemplate):
    def __init__(self, state, model_name):
        with open(os.path.join('./', 'models', model_name+'_model.pkl'), 'rb') as fd:
            self.model = pkl.load(fd)
        with open(os.path.join('./', 'models', model_name+'_token.pkl'), 'rb') as fd:
            self.tokenizer = pkl.load(fd)
        self.state = state
        
        if self.state['stem']:
            self.lemm = st.WordNetLemmatizer()
            self.stem = st.RSLPStemmer()

            
    def prepare_text(self, inp_text):
        if self.state['stem']:
            preprocessed = self.clean_text(inp_text)
            preprocessed = ' '.join([self.lemm.lemmatize(el) for el in preprocessed.split()])
            preprocessed = ' '.join([self.stem.stem(el) for el in preprocessed.split()])
            return preprocessed
        else:
            return self.clean_text(inp_text)
        
        
    def predict_tonality(self, inp_text):
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
    def __init__(self, state, model_name):
        #super().__init__(model_path)
        self.model = pipeline("sentiment-analysis", model=model_name)
        self.state = state
        
    def predict_tonality(self, inp_text):
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
