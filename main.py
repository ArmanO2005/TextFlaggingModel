import dask.dataframe as dd
# from dask_ml.feature_extraction.text import TfidfVectorizer
# from dask_ml.naive_bayes import MultinomialNB
# from dask_ml.model_selection import train_test_split


import pandas as pd
from gensim.models import Word2Vec
import gensim
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
import string

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


nltk.download('punkt_tab')
nltk.download('corpora')
nltk.download('wordnet')
nltk.download('taggers')  
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()


def Remove_punctuation(tokens):
    return [word for word in tokens if any([i in string.punctuation for i in word]) == False]


def Tokenize(data):
    for i, document in enumerate(data['Title']):
        data.loc[i, 'Title'] = word_tokenize(document.lower())


lemmatizer = WordNetLemmatizer()

def Lemmatize(data):
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    for i, document in enumerate(data['Title']):

        # check
        if type(data.loc[i, 'Title']) != list:
            print(data.loc[i, 'Title'])
            print(i)
        
        try:
            lemmatized_words = [lemmatizer.lemmatize(word, tag_map[tag[0]]) for word, tag in nltk.pos_tag(document) if word not in stopwords.words('english')]
            data.loc[i, 'Title'] = lemmatized_words
        except:
            print('error')
            print(document)
            break


def Join_tokens(data):
    for i, document in enumerate(data['Title']):
        data.loc[i, 'Title'] = ' '.join(document)
    

data = pd.read_csv("C://Users//arman//0 - Chat AI Stuff//training.csv")
# data = data.loc[data['Classification'] != 'Unsure']
# data = data.loc[data['Title'].str.split().str.len() > 1]


Tokenize(data)
Lemmatize(data)
Join_tokens(data)


tfidf_vect = TfidfVectorizer(max_features = 50000)
tfidf_matrix = tfidf_vect.fit_transform(data['Title'])

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vect.get_feature_names_out())

X = tfidf_df
y = data['Classification']


#----------------------------------------------
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#----------------------------------------------


model = MultinomialNB()
model.fit(X, y)


#----------------------------------------------
# y_pred = model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)

# print(f'Accuracy: {accuracy}')
# print(f'Classification Report:\n{report}')
#----------------------------------------------


joblib.dump(model, 'model.pkl')
joblib.dump(tfidf_vect, 'tfidf_vectorizer.joblib')