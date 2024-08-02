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

lemmetizer = WordNetLemmatizer()

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

data = pd.read_csv("C://Users//arman//Dropbox//1 - App Dev//Chat AI stuff//training.csv")

def remove_punctuation(tokens):
    return [word for word in tokens if word not in string.punctuation]

def tokenize(data):
    for i, text in enumerate(data['Title']):
        data.loc[i, 'Title'] = [word_tokenize(t) for t in sent_tokenize(text.lower())]

    for i, text in enumerate(data['Title']):
        for j, sentence in enumerate(text):
            sentence = remove_punctuation(sentence)
            data.loc[i, 'Title'][j] = [lemmetizer.lemmatize(word, tag_map[tag[0]]) for word, tag in nltk.pos_tag(sentence) if word not in stopwords.words('english')]
    
tokenize(data)
            
print(data)