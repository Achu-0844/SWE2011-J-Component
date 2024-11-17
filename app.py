from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

model = load_model('news_model.h5')
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

max_length = 20

def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences

def predict_headline(headline):
    processed_text = preprocess_text(headline)
    prediction = model.predict(processed_text)
    is_fake = 'Real' if prediction[0][0] > 0.5 else 'Fake'
    return is_fake

def get_author_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        author_patterns = [
            {'tag': 'a', 'attrs': {'class': 'BGiqL'}},
            {'tag': 'a', 'attrs': {'commonstate': '[object Object]'}},
            {'tag': 'div', 'attrs': {'class': 'author-name'}},
            {'tag': 'a', 'attrs': {'class': 'person-name lnk'}},
            {'tag': 'meta', 'attrs': {'name': 'author'}},
            {'tag': 'meta', 'attrs': {'property': 'article:author'}},
        ]

        author = None
        for pattern in author_patterns:
            author_elements = soup.find_all(pattern['tag'], attrs=pattern.get('attrs', {}))

            if author_elements:
                if pattern['tag'] == 'meta':
                    author = author_elements[0].get('content', '').strip()
                else:
                    author = author_elements[0].text.strip()
                break

        if not author:
            author = "No author found"
        
        return author

    except Exception as e:
        return f"Error extracting author: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON format"}), 400

    headline = data.get('headline', '')
    url = data.get('url', '')

    if not headline or not url:
        return jsonify({"error": "Both headline and URL must be provided."}), 400

    try:
        author = get_author_from_url(url)
    except Exception as e:
        return jsonify({"error": f"Error extracting author: {str(e)}"}), 500
    
    try:
        fake_news_prediction = predict_headline(headline)
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

    print(f"Headline: {headline}, URL: {url}, Author: {author}, Prediction: {fake_news_prediction}")

    return jsonify({
        "headline": headline,
        "url": url,
        "author": author,
        "prediction": fake_news_prediction
    })


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5001)
