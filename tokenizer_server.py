from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer
import requests
import json

app = Flask(__name__)

# Load tokenizer at startup
tokenizer = DistilBertTokenizer.from_pretrained('E1/model')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    
    # Tokenize with your original tokenizer
    tokens = tokenizer(
        text, 
        return_tensors="tf", 
        padding='max_length',
        truncation=True, 
        max_length=512
    )
    
    # Call TensorFlow Serving
    payload = {
    "instances": [
        {
            "input_ids": tokens["input_ids"].numpy().astype('int32')[0].tolist(),
            "attention_mask": tokens["attention_mask"].numpy().astype('int32')[0].tolist()
        }
    ]
    }

    
    response = requests.post(
        'http://localhost:8501/v1/models/fakenews_model:predict',
        json=payload
    )
    
    return response.json()

if __name__ == '__main__':
    app.run(port=5000, threaded=True)