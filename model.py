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

tfidf_vectorizer = TfidfVectorizer(
    max_df=0.90,
    ngram_range=(1,3), 
    min_df=2, 
    max_features=1500, 
    stop_words='english'
    )
rf=RandomForestClassifier(
    n_estimators=200,
    criterion='gini'
    )

data=pd.read_csv('Data/clean_data.csv')

tfidf_data=tfidf_vectorizer.fit_transform(data['cleanText'].apply(lambda x: np.str_(x)))

X_train,X_test,y_train,y_test=train_test_split(tfidf_data,data['sentiment'],test_size=0.3,random_state=0)

rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)

print(f1_score(y_test,y_pred,average='weighted'))

# save tfidf model and random forest model

filename_tfidf='Model/tfidf_1.sav'
filename_model='Model/rf_model_1.sav'

joblib.dump(tfidf_vectorizer,filename_tfidf)
joblib.dump(rf,filename_model)

print('done')