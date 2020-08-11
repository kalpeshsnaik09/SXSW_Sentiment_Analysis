import pandas as pd
import numpy as np

import re
from textblob import TextBlob
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score

import joblib

lemmatizer = WordNetLemmatizer()

#Create a function to clean tweets
def cleanText(text):
    text=str(text)  #Coverts Text to String
    text=re.sub(r'@[A-Za-z0-9]+','',text)  #Removing @Mentions
    text = re.sub(r'#[\w]*sxsw[\w]*', ' ', text,flags=re.I)  #Removing sxsw hashtag
    text=re.sub(r'#','',text)  #Removing # Symbols
    text=re.sub(r'RT[\s]+','',text)  #Removing ReTweets
    text=re.sub(r'https?:\/\/\s+','',text)  #Removing the hyperlinks
    text=re.sub(r'bit.ly[/\.\w]+','',text)  #Removing the shortlinks 
    text=text.replace(r'{html}',"") 
    cleanr = re.compile(r'<.*?>')
    text = re.sub(cleanr, '', text)
    text = re.sub(r'[0-9]+', '', text)  #Removing Numbers
    text = re.sub(r'[^A-Za-z]+', ' ', text)  #Removing all spacial character
    text = text.lower()  #Coverts Text To Lower Case
    
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text) 
    filtered_words = [w for w in tokens if w not in stopwords.words('english')]
    lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words)

class model_predict:
    def __init__(self,input_dir_name,output_dir_name):
        self.input_dir_name=input_dir_name
        self.output_dir_name=output_dir_name
        data=pd.read_csv(input_dir_name)
        data['cleanText']=data['tweet'].map(lambda s:cleanText(s))
        tfidf_model=joblib.load('Model/tfidf_1.sav')
        data_tfidf=tfidf_model.transform(data['cleanText'])
        model=joblib.load('Model/rf_model_1.sav')
        y_pred=model.predict(data_tfidf)
        output=pd.concat([data['tweet_id'],pd.DataFrame(y_pred,columns=['sentiment'])],axis=1)
        output.to_csv(output_dir_name,index=False)

if __name__ == "__main__":
    input_dir=input('Enter input file path : ')
    output_dir=input('Enter output file path : ')
    model_predict(input_dir,output_dir)