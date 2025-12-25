import os

import torch

from src.data_preprocessing import (
    normalizeString,
    tensorFromSentence,
    SOS_token,
    EOS_token,
    Lang,
)
from src.models import EncoderRNN, DecoderRNN, Seq2Seq, AttnDecoderRNN, Seq2SeqAttn


MODEL_DIR = "models"
CHECKPOINT_NAME = "seq2seq_en_fr.pt"


def load_model(checkpoint_path=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        checkpoint_path = os.path.join(MODEL_DIR, CHECKPOINT_NAME)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            f"Train the model first by running train.py."
        )

    # Handle PyTorch 2.6+ safe loading (weights_only=True by default)
    try:
        from torch.serialization import add_safe_globals

        # Allowlist Lang so the checkpoint (which stores Lang objects)
        # can be safely unpickled.
        add_safe_globals([Lang])
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception:
        # Fallback for older PyTorch versions or if safe loading fails:
        # explicitly disable weights_only to allow full unpickling.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    input_lang = checkpoint["input_lang"]
    output_lang = checkpoint["output_lang"]
    config = checkpoint["config"]

    use_attention = bool(config.get("use_attention", False))
    enc_bidirectional = bool(config.get("enc_bidirectional", False))

    encoder = EncoderRNN(
        input_vocab_size=input_lang.n_words,
        embedding_dim=config["embedding_dim"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        bidirectional=enc_bidirectional,
    ).to(device)

    if use_attention:
        enc_output_dim = config["hidden_size"] * (2 if enc_bidirectional else 1)
        decoder = AttnDecoderRNN(
            output_vocab_size=output_lang.n_words,
            embedding_dim=config["embedding_dim"],
            hidden_size=config["hidden_size"],
            enc_output_dim=enc_output_dim,
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        ).to(device)
    else:
        decoder = DecoderRNN(
            output_vocab_size=output_lang.n_words,
            embedding_dim=config["embedding_dim"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        ).to(device)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    if use_attention:
        model = Seq2SeqAttn(encoder, decoder, device, enc_bidirectional=enc_bidirectional).to(device)
    else:
        model = Seq2Seq(encoder, decoder, device).to(device)
    model.eval()

    return model, input_lang, output_lang, device


def translate_sentence(sentence, model, input_lang, output_lang, device, max_length=30):
    model.eval()

    normalized = normalizeString(sentence)
    src_tensor = tensorFromSentence(input_lang, normalized).to(device)

    src_length = [src_tensor.size(0)]
    src_tensor = src_tensor.unsqueeze(1)  # (seq_len, 1)

    with torch.no_grad():
        if isinstance(model, Seq2SeqAttn):
            encoder_outputs, enc_state = model.encoder(src_tensor, src_length)
            hidden, cell = model._init_dec_state(enc_state)
            input_token = torch.tensor([SOS_token], dtype=torch.long, device=device)

            decoded_tokens = []
            for _ in range(max_length):
                output, hidden, cell = model.decoder(
                    input_token, hidden, cell, encoder_outputs, src_length
                )
                top1 = output.argmax(1)
                token_id = top1.item()
                if token_id == EOS_token:
                    break
                decoded_tokens.append(token_id)
                input_token = top1
        else:
            _, (hidden, cell) = model.encoder(src_tensor, src_length)

            input_token = torch.tensor([SOS_token], dtype=torch.long, device=device)

            decoded_tokens = []

            for _ in range(max_length):
                output, hidden, cell = model.decoder(input_token, hidden, cell)
                top1 = output.argmax(1)
                token_id = top1.item()

                if token_id == EOS_token:
                    break

                decoded_tokens.append(token_id)
                input_token = top1

    translated_words = [output_lang.index2word.get(idx, "<UNK>") for idx in decoded_tokens]
    return " ".join(translated_words)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Translate English -> French using a trained checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint .pt file. Defaults to models/seq2seq_en_fr.pt",
    )
    args = parser.parse_args()

    model, input_lang, output_lang, device = load_model(checkpoint_path=args.checkpoint)

    print("Enter an English sentence to translate (empty line to quit):")
    while True:
        sentence = input("> ").strip()
        if not sentence:
            break

        translation = translate_sentence(sentence, model, input_lang, output_lang, device)
        print(f"French: {translation}")


if __name__ == "__main__":
    main()
