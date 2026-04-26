
# Fake News Detector 📰

A machine learning system designed to identify and classify potentially false or misleading information.

## Dataset 📊
- Source: Kaggle
- Features include:
  - Headline
  - News

## Objectives 🎯

- Perform data cleaning and preprocessing
- News classification and analysis
- Conduct exploratory data analysis (EDA)
- Build and evaluate multiple classification models

## Tech Stack 🛠️
- Python
- NumPy
- Pandas
- Matplotlib / Seaborn
- Scikit-learn
- Jupyter Notebook


## Workflow 🏗️
1. Data Cleaning (handling missing values)
2. NLP based text preprocessing:
    - Removal of HTML tags and URLs
    - Removal of puntuation marks
    - Tokenization
    - Stop words removal
    - Stemming
    - Vectorization using Bag of n-grams (bag of words, bi-grams, tri-grams)
3. 
4. Model Building:
   - Logistic Regression
   - Multiomial Naive Bayes
5. Model Evaluation (F1-Score, precision score and accuracy score)


## Results 📈
- Best Model: Logistic Regression using Bag-of-bigrams vectorizer
- F1-Score: 96%
- Precision: 97% for real news and 95% for fake news

