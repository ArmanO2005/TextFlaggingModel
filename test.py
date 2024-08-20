import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
import string
from main import Tokenize, Lemmatize, Join_tokens, Remove_punctuation

nltk.download('punkt')
nltk.download('corpora')
nltk.download('wordnet')
nltk.download('taggers')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

vectorizer = joblib.load('tfidf_vectorizer.joblib')
trained_model = joblib.load('model.pkl')

def Format_string(string):
    x = pd.DataFrame({'Title': [string]})
    Remove_punctuation(x)
    Tokenize(x)
    Lemmatize(x)
    Join_tokens(x)
    return vectorizer.transform(x['Title'])

while True:
    test_document = input("message:")

    formatted_test_doc = Format_string(test_document)
    prediction = trained_model.predict(formatted_test_doc)

    print(prediction)