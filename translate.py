import os

import torch

from src.data_preprocessing import (
    normalizeString,
    normalizeString_v2,
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
    model.text_normalization = str(config.get("text_normalization", "v1"))

    return model, input_lang, output_lang, device


def translate_sentence(sentence, model, input_lang, output_lang, device, max_length=30):
    return translate_sentence_beam(
        sentence,
        model,
        input_lang,
        output_lang,
        device,
        max_length=max_length,
        beam_size=1,
        length_penalty=0.0,
    )


def _length_penalized_score(log_prob, length, alpha):
    if alpha <= 0:
        return log_prob
    length = max(1, length)
    return log_prob / (float(length) ** float(alpha))


@torch.no_grad()
def translate_sentence_beam(
    sentence,
    model,
    input_lang,
    output_lang,
    device,
    max_length=30,
    beam_size=5,
    length_penalty=0.6,
):
    model.eval()

    normalizer = normalizeString_v2 if getattr(model, "text_normalization", "v1") == "v2" else normalizeString
    normalized = normalizer(sentence)
    src_tensor = tensorFromSentence(input_lang, normalized).to(device)
    src_length = [src_tensor.size(0)]
    src_tensor = src_tensor.unsqueeze(1)  # (seq_len, 1)

    if beam_size <= 1:
        beam_size = 1

    if isinstance(model, Seq2SeqAttn):
        encoder_outputs, enc_state = model.encoder(src_tensor, src_length)
        hidden, cell = model._init_dec_state(enc_state)

        beams = [([SOS_token], 0.0, hidden, cell, None, False)]

        for _ in range(max_length):
            candidates = []
            for tokens, score, hidden, cell, context, ended in beams:
                if ended:
                    candidates.append((tokens, score, hidden, cell, context, ended))
                    continue

                input_token = torch.tensor([tokens[-1]], dtype=torch.long, device=device)
                logits, next_hidden, next_cell, next_context = model.decoder(
                    input_token, hidden, cell, encoder_outputs, src_length, context
                )
                log_probs = torch.log_softmax(logits, dim=1)  # (1, vocab)
                top_log_probs, top_ids = log_probs.topk(beam_size, dim=1)

                for lp, idx in zip(top_log_probs.squeeze(0).tolist(), top_ids.squeeze(0).tolist()):
                    new_tokens = tokens + [idx]
                    new_score = score + float(lp)
                    new_ended = idx == EOS_token
                    candidates.append(
                        (new_tokens, new_score, next_hidden, next_cell, next_context, new_ended)
                    )

            beams = sorted(
                candidates,
                key=lambda b: _length_penalized_score(b[1], len(b[0]) - 1, length_penalty),
                reverse=True,
            )[:beam_size]

            if all(ended for *_, ended in beams):
                break

        best_tokens = max(
            beams,
            key=lambda b: _length_penalized_score(b[1], len(b[0]) - 1, length_penalty),
        )[0]
        decoded_tokens = [t for t in best_tokens[1:] if t != EOS_token]
    else:
        _, (hidden, cell) = model.encoder(src_tensor, src_length)

        beams = [([SOS_token], 0.0, hidden, cell, False)]

        for _ in range(max_length):
            candidates = []
            for tokens, score, hidden, cell, ended in beams:
                if ended:
                    candidates.append((tokens, score, hidden, cell, ended))
                    continue

                input_token = torch.tensor([tokens[-1]], dtype=torch.long, device=device)
                logits, next_hidden, next_cell = model.decoder(input_token, hidden, cell)
                log_probs = torch.log_softmax(logits, dim=1)
                top_log_probs, top_ids = log_probs.topk(beam_size, dim=1)

                for lp, idx in zip(top_log_probs.squeeze(0).tolist(), top_ids.squeeze(0).tolist()):
                    new_tokens = tokens + [idx]
                    new_score = score + float(lp)
                    new_ended = idx == EOS_token
                    candidates.append((new_tokens, new_score, next_hidden, next_cell, new_ended))

            beams = sorted(
                candidates,
                key=lambda b: _length_penalized_score(b[1], len(b[0]) - 1, length_penalty),
                reverse=True,
            )[:beam_size]

            if all(ended for *_, ended in beams):
                break

        best_tokens = max(
            beams,
            key=lambda b: _length_penalized_score(b[1], len(b[0]) - 1, length_penalty),
        )[0]
        decoded_tokens = [t for t in best_tokens[1:] if t != EOS_token]

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
    parser.add_argument("--max-length", type=int, default=30, help="Maximum decoded length (tokens).")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size; use 1 for greedy decoding.")
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=0.6,
        help="Length penalty alpha (0 disables). Higher discourages short outputs.",
    )
    args = parser.parse_args()

    model, input_lang, output_lang, device = load_model(checkpoint_path=args.checkpoint)

    print("Enter an English sentence to translate (empty line to quit):")
    while True:
        sentence = input("> ").strip()
        if not sentence:
            break

        translation = translate_sentence_beam(
            sentence,
            model,
            input_lang,
            output_lang,
            device,
            max_length=args.max_length,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
        )
        print(f"French: {translation}")


if __name__ == "__main__":
    main()
