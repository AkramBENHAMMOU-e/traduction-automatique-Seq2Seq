import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from translate import load_model as load_seq2seq_model
from translate import translate_sentence as translate_with_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "seq2seq_en_fr.pt")
MAX_LENGTH = 20

# Global variables for model
model = None
input_lang = None
output_lang = None

def load_model():
    global model, input_lang, output_lang
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    print(f"Loading model from {MODEL_PATH}...")
    model_loaded, in_lang, out_lang, _device = load_seq2seq_model(
        checkpoint_path=MODEL_PATH, device=device
    )
    model = model_loaded
    input_lang = in_lang
    output_lang = out_lang
    print("Model loaded successfully!")

def translate_sentence(sentence):
    return translate_with_model(
        sentence,
        model=model,
        input_lang=input_lang,
        output_lang=output_lang,
        device=device,
        max_length=MAX_LENGTH,
    )

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': MODEL_PATH})

@app.route('/translate', methods=['POST'])
def translate_api():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
        
    source_text = data.get('text', '')
    
    try:
        translation = translate_sentence(source_text)
        return jsonify({'translation': translation})
    except Exception as e:
        print(f"Error processing {source_text}: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    print(f"Server running on port 5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
