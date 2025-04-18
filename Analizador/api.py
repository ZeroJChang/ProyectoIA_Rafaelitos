from flask import Flask, request, jsonify
import pickle
import os
import re
from flask_cors import CORS 
import math

app = Flask(__name__)
CORS(app)

# Cargar modelo
model_path = os.path.join(os.path.dirname(__file__), 'bbc_classifier.pkl')
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)
    class_probs = model_data['class_probs']
    word_probs = model_data['word_probs']
    vocabulary = set(model_data['vocabulary'])
    stopwords = model_data['stopwords']

def preprocess_text(text):
    """Preprocesamiento consistente con el entrenamiento"""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    return [word for word in words if word not in stopwords and len(word) > 2]

@app.route('/classify', methods=['POST'])
def classify():
    import math

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Se requiere texto para clasificar'}), 400

    text = data['text']
    tokens = preprocess_text(text)
    features = {word: 1 if word in vocabulary else 0 for word in tokens}

    category_scores = {}

    for category in class_probs:
        log_prob = math.log(class_probs[category])
        for word, count in features.items():
            if count > 0 and word in word_probs.get(category, {}):
                log_prob += math.log(word_probs[category][word]) * count
        category_scores[category] = log_prob

    log_sum = max(category_scores.values())
    exp_scores = {cat: math.exp(score - log_sum) for cat, score in category_scores.items()}
    total = sum(exp_scores.values())
    probs = {cat: round((score / total) * 100, 2) for cat, score in exp_scores.items()}

    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    result = {
        "categories": [
            {"category": cat, "confidence": conf}
            for cat, conf in sorted_probs
        ],
        "status": "success"
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)