import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import normalizeString, tensorFromSentence, SOS_token, EOS_token, Lang
from src.models import EncoderRNN, DecoderRNN, Seq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

MODEL_PATH = 'models/seq2seq_en_fr.pt'
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
    # Load checkpoint
    # weights_only=False is needed because the checkpoint contains custom Lang objects
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    input_lang = checkpoint['input_lang']
    output_lang = checkpoint['output_lang']
    
    config = checkpoint.get('config', {})
    embedding_dim = config.get('embedding_dim', 256)
    hidden_size = config.get('hidden_size', 256)
    num_layers = config.get('num_layers', 1)
    dropout_p = config.get('dropout', 0.1)
    
    print(f"Model Config: {config}")

    # Initialize models using the architecture from src.models (LSTM based)
    encoder = EncoderRNN(input_lang.n_words, embedding_dim, hidden_size, num_layers, dropout_p).to(device)
    decoder = DecoderRNN(output_lang.n_words, embedding_dim, hidden_size, num_layers, dropout_p).to(device)
    
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    # Load state dicts
    if 'model_state_dict' in checkpoint:
         model.load_state_dict(checkpoint['model_state_dict'])
    else:
         model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
         model.decoder.load_state_dict(checkpoint['decoder_state_dict'])

    model.eval()
    print("Model loaded successfully!")

def translate_sentence(sentence):
    with torch.no_grad():
        sentence = normalizeString(sentence)
        try:
            input_tensor = tensorFromSentence(input_lang, sentence).to(device) # (seq_len,)
        except Exception as e:
            return f"Error: Could not encode sentence. {str(e)}"

        # Prepare input for model: (seq_len, batch_size)
        input_tensor = input_tensor.unsqueeze(1) 
        src_lengths = [len(input_tensor)]
        
        # Encode
        _, (hidden, cell) = model.encoder(input_tensor, src_lengths)
        
        # Prepare decoder input (SOS token)
        decoder_input = torch.tensor([SOS_token], device=device) # (batch_size,)
        
        decoded_words = []
        
        for di in range(MAX_LENGTH):
            # Decoder forward expects: (input_step, hidden, cell)
            # input_step: (batch_size,)
            output, hidden, cell = model.decoder(decoder_input, hidden, cell)
            
            topv, topi = output.topk(1)
            token_id = topi.squeeze().item()
            
            if token_id == EOS_token:
                break
            else:
                word = output_lang.index2word.get(token_id, '<UNK>')
                decoded_words.append(word)
            
            # Next input is current prediction
            decoder_input = topi.view(-1).detach() # (batch_size,)
            
        return ' '.join(decoded_words)

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
