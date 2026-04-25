#!/usr/bin/env python
# coding: utf-8

# ## Import necessary libs

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Load and get info on dataset 

# In[2]:


data=pd.read_csv("../WELFake_Dataset.csv")


# In[3]:


data.head()


# In[4]:


data.set_index(data.columns[0], inplace=True)


# In[5]:


data.info()


# In[6]:


data.isnull().sum()


# In[7]:


data['label'].value_counts()

1=ham
0=spam
# In[8]:


data = data.dropna()
data


# In[9]:


data.info()


# ## Remove HTML tags and URLs

# In[10]:


import re
def remove_tags(text):
    pattern = re.compile(r'<.*?>')
    return pattern.sub(r'', text)
def remove_links(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)


# In[11]:


data['title'] = data['title'].apply(remove_tags)
data['text'] = data['text'].apply(remove_tags)
data['title'] = data['title'].apply(remove_links)
data['text'] = data['text'].apply(remove_links)
data.head()


# ## Convert dataset into lowercase

# In[12]:


data['title'] = data['title'].str.lower()
data['text'] = data['text'].str.lower()
data.head()


# ## Remove punctuation marks

# In[13]:


import string
pm = string.punctuation
def remove_punctuation(text):
    pattern = re.compile(r'[^\w\s]')
    return pattern.sub(r' ', text)


# In[14]:


data['title'] = data['title'].apply(remove_punctuation)
data['text'] = data['text'].apply(remove_punctuation)


# In[15]:


data.head()


# ## Tokenize the dataset

# In[16]:


from nltk.tokenize import word_tokenize
data['title_tokens'] = data['title'].apply(lambda x: word_tokenize(str(x)))
data['text_tokens'] = data['text'].apply(lambda x: word_tokenize(str(x)))
data.head()


# ## Remove stop words

# In[17]:


from nltk.corpus import stopwords
sw = stopwords.words('english')
sw


# In[18]:


def remove_stopwords(text):
    new_text = []
    for word in text:
        if word in sw:
            new_text.append('')
        else:
            new_text.append(word)
    return " ".join(new_text)


# In[19]:


data['title_tokens'] = data['title_tokens'].apply(remove_stopwords)
data['text_tokens'] = data['text_tokens'].apply(remove_stopwords)
data.head()


# ## Stemming

# In[20]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem_words(text):
    return " ".join([ps.stem(word) for word in text.split()])


# In[21]:


data['title_tokens'] = data['title_tokens'].apply(stem_words)
data['text_tokens'] = data['text_tokens'].apply(stem_words)
data


# In[22]:


y = data['label']
y


# ## Separate preprocessed data

# In[23]:


data['clean_text'] = data['title_tokens'] + " " + data['text_tokens']
data


# ## Train Test Split

# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(data['clean_text'],y, test_size = 0.3, random_state=23)


# ## Vectorize data using various vectorizers

# In[31]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
cvb = CountVectorizer(ngram_range=(2,2)) # bigram
cvt = CountVectorizer(max_features=50000, ngram_range=(3,3)) # trigram
tfidf = TfidfVectorizer()


# In[32]:


cv_encoded_train= cv.fit_transform(X_train)
cvb_encoded_train= cvb.fit_transform(X_train)
cvt_encoded_train= cvt.fit_transform(X_train)
tfidf_encoded_train= tfidf.fit_transform(X_train)


# In[33]:


print(len(cv.vocabulary_))
print(len(cvb.vocabulary_))
print(len(cvt.vocabulary_))
print(len(tfidf.vocabulary_))


# In[34]:


cv_encoded_test= cv.transform(X_test)
cvb_encoded_test= cvb.transform(X_test)
cvt_encoded_test= cvt.transform(X_test)
tfidf_encoded_test= tfidf.transform(X_test)


# ## Train Logistic model

# In[35]:


from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression(max_iter=5000)
model2 = LogisticRegression(max_iter=5000)
model3 = LogisticRegression(max_iter=5000)
model4 = LogisticRegression(max_iter=5000)
model1.fit(cv_encoded_train, y_train)
model2.fit(cvb_encoded_train, y_train)
model3.fit(cvt_encoded_train, y_train)
model4.fit(tfidf_encoded_train, y_train)


# In[36]:


print(model1.score(cv_encoded_test, y_test))
print(model2.score(cvb_encoded_test, y_test))
print(model3.score(cvt_encoded_test, y_test))
print(model4.score(tfidf_encoded_test, y_test))


# In[37]:


y_pred_cv=model1.predict(cv_encoded_test)
y_pred_cvb=model2.predict(cvb_encoded_test)
y_pred_cvt=model3.predict(cvt_encoded_test)
y_pred_tfidf=model4.predict(tfidf_encoded_test)


# ### The Logistic regression model yielded the best result when the input data was vectorized using bag of bi-grams.

# ## Train Naive Bayes' Model

# In[38]:


from sklearn.naive_bayes import MultinomialNB
nb1 = MultinomialNB()
nb2 = MultinomialNB()
nb3 = MultinomialNB()
nb4 = MultinomialNB()
nb1


# In[39]:


nb1.fit(cv_encoded_train, y_train)
nb2.fit(cvb_encoded_train, y_train)
nb3.fit(cvt_encoded_train, y_train)
nb4.fit(tfidf_encoded_train, y_train)


# In[40]:


print(nb1.score(cv_encoded_test, y_test))
print(nb2.score(cvb_encoded_test, y_test))
print(nb3.score(cvt_encoded_test, y_test))
print(nb4.score(tfidf_encoded_test, y_test))


# In[41]:


y1_pred_cv=nb1.predict(cv_encoded_test)
y1_pred_cvb=nb2.predict(cvb_encoded_test)
y1_pred_cvt=nb3.predict(cvt_encoded_test)
y1_pred_tfidf=nb4.predict(tfidf_encoded_test)


# ### The Multinomial NB model yielded the best result when the input data was vectorized using bag of bi-grams.

# ## Get classification report

# In[42]:


from sklearn.metrics import classification_report
print(classification_report(y_pred_cvb, y_test))
print(classification_report(y1_pred_cvb, y_test))


# ### 

# ## make a pipeline using vectorizer and model with the highest scores(bi-grams and Logistic Regression model)

# In[43]:


from sklearn.pipeline import Pipeline
pipe = Pipeline([('vectorizer', CountVectorizer(max_features=70000, ngram_range=(2,2))),
    ('model', LogisticRegression(max_iter=5000))])
pipe


# In[44]:


pipe.fit(X_train,y_train)


# In[45]:


pipe.score(X_test,y_test)

