import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import gensim
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
import string

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('corpora')
nltk.download('wordnet')
nltk.download('taggers')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

def remove_punctuation(tokens):
    return [word for word in tokens if any([i in string.punctuation for i in word]) == False]


def tokenize(data):
    for i, document in enumerate(data['Title']):
        data.loc[i, 'Title'] = [word_tokenize(t) for t in sent_tokenize(document.lower())]
    
    for i, document in enumerate(data['Title']):    
        data.loc[i, 'Title'] = [remove_punctuation(sentence) for sentence in document]

def lemmatize(data):
    for i, document in enumerate(data['Title']):
        for j, sentence in enumerate(document):
            sentence = remove_punctuation(sentence)
            data.loc[i, 'Title'][j] = [lemmatizer.lemmatize(word, tag_map[tag[0]]) for word, tag in nltk.pos_tag(sentence) if word not in stopwords.words('english')]
    
def join_tokens(data):
    for i, text in enumerate(data['Title']):
        data.loc[i, 'Title'] = ' '.join([' '.join(sentence) for sentence in text])
    

data = pd.read_csv("C://Users//arman//Dropbox//1 - App Dev//Chat AI stuff//training.csv").dropna()
tokenize(data)

data.to_csv('output.csv', index=False)


# tfidf_vect = TfidfVectorizer(max_features=5000)
# tfidf_matrix = tfidf_vect.fit_transform(data['Title'])

# tfidf_vect.get_feature_names_out()

# tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vect.get_feature_names_out())

# tfidf_df.to_csv('output.csv', index=False)

# X_train, X_test, y_train, y_test = model_selection.train_test_split(tfidf_matrix, data['Label'], test_size=0.3)


# svm_model = svm.SVC(kernel='linear')
# svm_model.fit(X_train, y_train)


# predictions = svm_model.predict(X_test)
# accuracy = accuracy_score(y_test, predictions)
# print(f"Accuracy: {accuracy}")