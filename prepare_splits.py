import argparse
import json
from pathlib import Path

from src.data_preprocessing import load_pairs


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test EN/FR files from the Tatoeba/ManyThings TSV.")
    parser.add_argument("--data", type=str, default="data/tatoeba/fra.txt", help="Input TSV/CSV with EN/FR pairs.")
    parser.add_argument("--out-dir", type=str, default="data/splits", help="Output directory for split files.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=15, help="Max tokens per sentence (after normalization).")
    parser.add_argument("--norm", type=str, default="v2", choices=["v1", "v2"])
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of pairs (debug).")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    args = parser.parse_args()

    if not (0.0 <= args.val_split < 1.0) or not (0.0 <= args.test_split < 1.0):
        raise SystemExit("val-split and test-split must be in [0, 1).")
    if args.val_split + args.test_split >= 1.0:
        raise SystemExit("val-split + test-split must be < 1.0.")

    pairs = load_pairs(
        args.data,
        limit=args.limit,
        max_length=args.max_length,
        normalization=args.norm,
    )
    if not pairs:
        raise SystemExit("No sentence pairs loaded (check --data, --max-length, and --norm).")

    import torch

    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(pairs), generator=g).tolist()
    pairs = [pairs[i] for i in idx]

    n_total = len(pairs)
    n_test = int(n_total * args.test_split)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_test - n_val

    test_pairs = pairs[:n_test]
    val_pairs = pairs[n_test : n_test + n_val]
    train_pairs = pairs[n_test + n_val :]

    out_dir = Path(args.out_dir)
    _write_lines(out_dir / "train.en", [p[0] for p in train_pairs])
    _write_lines(out_dir / "train.fr", [p[1] for p in train_pairs])
    _write_lines(out_dir / "val.en", [p[0] for p in val_pairs])
    _write_lines(out_dir / "val.fr", [p[1] for p in val_pairs])
    _write_lines(out_dir / "test.en", [p[0] for p in test_pairs])
    _write_lines(out_dir / "test.fr", [p[1] for p in test_pairs])

    meta = {
        "source": args.data,
        "out_dir": str(out_dir),
        "seed": args.seed,
        "max_length": args.max_length,
        "normalization": args.norm,
        "limit": args.limit,
        "splits": {
            "train": n_train,
            "val": n_val,
            "test": n_test,
            "total": n_total,
        },
    }
    _write_lines(out_dir / "meta.json", [json.dumps(meta, indent=2, ensure_ascii=False)])

    print("Wrote splits to", out_dir)
    print(f"train: {n_train} | val: {n_val} | test: {n_test} | total: {n_total}")


if __name__ == "__main__":
    main()

