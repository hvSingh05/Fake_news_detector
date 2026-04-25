import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import nltk
import os

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

print("Loading data...")
data=pd.read_csv("WELFake_Dataset.csv", nrows=20000)
data.set_index(data.columns[0], inplace=True)
data = data.dropna()

def remove_tags(text):
    pattern = re.compile(r'<.*?>')
    return pattern.sub(r'', text)

def remove_links(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

def remove_punctuation(text):
    pattern = re.compile(r'[^\w\s]')
    return pattern.sub(r' ', text)

print("Preprocessing...")
data['title'] = data['title'].apply(remove_tags).apply(remove_links).str.lower().apply(remove_punctuation)
data['text'] = data['text'].apply(remove_tags).apply(remove_links).str.lower().apply(remove_punctuation)

sw = set(stopwords.words('english'))
ps = PorterStemmer()

def process_text(text):
    tokens = word_tokenize(str(text))
    return " ".join([ps.stem(word) for word in tokens if word not in sw])

print("Tokenizing and Stemming...")
data['title'] = data['title'].apply(process_text)
data['text'] = data['text'].apply(process_text)

data['clean_text'] = data['title'] + " " + data['text']
y = data['label']

print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(data['clean_text'], y, test_size=0.3, random_state=23)

pipe = Pipeline([
    ('vectorizer', CountVectorizer(max_features=10000, ngram_range=(1,1))),
    ('model', LogisticRegression(max_iter=5000))
])

pipe.fit(X_train, y_train)
print("Score:", pipe.score(X_test, y_test))

joblib.dump(pipe, 'fake_news_model.joblib')
print("Model saved to fake_news_model.joblib")
