import argparse
import math

import torch

from src.data_preprocessing import prepareData, normalizeString
from translate import load_model


def _ngram_counts(tokens, n):
    counts = {}
    if len(tokens) < n:
        return counts
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i : i + n])
        counts[ng] = counts.get(ng, 0) + 1
    return counts


def corpus_bleu(references, hypotheses, max_n=4, smooth=1.0):
    """
    references: list[list[str]] where inner list is reference tokens
    hypotheses: list[list[str]] where inner list is hypothesis tokens
    """
    clipped = [0] * max_n
    total = [0] * max_n
    ref_len = 0
    hyp_len = 0

    for ref, hyp in zip(references, hypotheses):
        ref_len += len(ref)
        hyp_len += len(hyp)
        for n in range(1, max_n + 1):
            hyp_counts = _ngram_counts(hyp, n)
            ref_counts = _ngram_counts(ref, n)
            total[n - 1] += max(0, len(hyp) - n + 1)
            for ng, c in hyp_counts.items():
                clipped[n - 1] += min(c, ref_counts.get(ng, 0))

    precisions = []
    for n in range(1, max_n + 1):
        p = (clipped[n - 1] + smooth) / (total[n - 1] + smooth)
        precisions.append(p)

    if hyp_len == 0:
        return 0.0
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - (ref_len / max(1, hyp_len)))
    score = bp * math.exp(sum(math.log(p) for p in precisions) / max_n)
    return 100.0 * score


@torch.no_grad()
def greedy_translate(sentence, model, input_lang, output_lang, device, max_length=30):
    from src.data_preprocessing import tensorFromSentence, SOS_token, EOS_token

    model.eval()
    src_tensor = tensorFromSentence(input_lang, normalizeString(sentence)).to(device)
    src_length = [src_tensor.size(0)]
    src_tensor = src_tensor.unsqueeze(1)  # (src_len, 1)

    if model.__class__.__name__ == "Seq2SeqAttn":
        encoder_outputs, enc_state = model.encoder(src_tensor, src_length)
        hidden, cell = model._init_dec_state(enc_state)
        input_token = torch.tensor([SOS_token], dtype=torch.long, device=device)
        decoded = []
        for _ in range(max_length):
            logits, hidden, cell = model.decoder(
                input_token, hidden, cell, encoder_outputs, src_length
            )
            top1 = logits.argmax(1)
            token_id = top1.item()
            if token_id == EOS_token:
                break
            decoded.append(output_lang.index2word.get(token_id, "<UNK>"))
            input_token = top1
        return " ".join(decoded)

    _, (hidden, cell) = model.encoder(src_tensor, src_length)
    input_token = torch.tensor([SOS_token], dtype=torch.long, device=device)
    decoded = []
    for _ in range(max_length):
        logits, hidden, cell = model.decoder(input_token, hidden, cell)
        top1 = logits.argmax(1)
        token_id = top1.item()
        if token_id == EOS_token:
            break
        decoded.append(output_lang.index2word.get(token_id, "<UNK>"))
        input_token = top1
    return " ".join(decoded)


def split_pairs(pairs, test_split, seed=42):
    if not (0.0 < test_split < 1.0):
        raise ValueError("test_split must be in (0.0, 1.0)")
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(pairs), generator=g).tolist()
    test_size = int(len(pairs) * test_split)
    test_idx = set(idx[:test_size])
    test_pairs = [pairs[i] for i in range(len(pairs)) if i in test_idx]
    return test_pairs


def score_checkpoint(ckpt_path, test_pairs, max_length):
    model, input_lang, output_lang, device = load_model(checkpoint_path=ckpt_path)
    refs = []
    hyps = []
    for src, ref in test_pairs:
        hyp = greedy_translate(src, model, input_lang, output_lang, device, max_length=max_length)
        refs.append(ref.split())
        hyps.append(hyp.split())
    bleu = corpus_bleu(refs, hyps)
    avg_len = sum(len(h) for h in hyps) / max(1, len(hyps))
    return bleu, avg_len


def main():
    parser = argparse.ArgumentParser(description="Compare two checkpoints on the same test split.")
    parser.add_argument("--ckpt-a", type=str, required=True)
    parser.add_argument("--ckpt-b", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/eng_-french.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=30)
    args = parser.parse_args()

    _, _, pairs = prepareData(args.data, limit=args.limit)
    test_pairs = split_pairs(pairs, test_split=args.test_split, seed=args.seed)
    print(f"Test samples: {len(test_pairs)}")

    bleu_a, len_a = score_checkpoint(args.ckpt_a, test_pairs, max_length=args.max_length)
    bleu_b, len_b = score_checkpoint(args.ckpt_b, test_pairs, max_length=args.max_length)

    print(f"A: {args.ckpt_a}")
    print(f"  BLEU: {bleu_a:.2f} | avg_hyp_len: {len_a:.1f}")
    print(f"B: {args.ckpt_b}")
    print(f"  BLEU: {bleu_b:.2f} | avg_hyp_len: {len_b:.1f}")


if __name__ == "__main__":
    main()
