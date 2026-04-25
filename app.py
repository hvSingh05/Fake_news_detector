import os
import re
import string
import joblib
from flask import Flask, render_template, request, jsonify
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

app = Flask(__name__)

# Ensure NLTK data is available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Load Model
MODEL_PATH = 'fake_news_model.joblib'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

# Preprocessing tools
sw = set(stopwords.words('english'))
ps = PorterStemmer()

def remove_tags(text):
    pattern = re.compile(r'<.*?>')
    return pattern.sub(r'', str(text))

def remove_links(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', str(text))

def remove_punctuation(text):
    pattern = re.compile(r'[^\w\s]')
    return pattern.sub(r' ', str(text))

def process_text(text):
    text = remove_tags(text)
    text = remove_links(text)
    text = text.lower()
    text = remove_punctuation(text)
    tokens = word_tokenize(text)
    return " ".join([ps.stem(word) for word in tokens if word not in sw])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not trained yet. Please wait a moment.'}), 503
        
    data = request.json
    title = data.get('headline', '')
    text = data.get('news', '')
    
    if not title and not text:
        return jsonify({'error': 'Please provide headline or news content.'}), 400
        
    processed_title = process_text(title)
    processed_text = process_text(text)
    
    clean_text = processed_title + " " + processed_text
    
    # 1 is Fake, 0 is Real based on WELFake dataset
    prediction = model.predict([clean_text])[0]
    
    result = 'Fake' if prediction == 1 else 'Real'
    
    return jsonify({
        'prediction': result,
        'clean_text': clean_text
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
