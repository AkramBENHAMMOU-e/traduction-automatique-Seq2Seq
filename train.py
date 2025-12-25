import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.data_preprocessing import (
    prepareData,
    TranslationDataset,
    collate_fn,
    PAD_token,
)
from src.models import EncoderRNN, DecoderRNN, Seq2Seq, AttnDecoderRNN, Seq2SeqAttn

try:
    import mlflow
except ImportError:
    mlflow = None


DATA_PATH = "data/eng_-french.csv"
MODEL_DIR = "models"
CHECKPOINT_NAME = "seq2seq_en_fr.pt"

EMBEDDING_DIM = 256
HIDDEN_SIZE = 256
NUM_LAYERS = 1
DROPOUT = 0.1

BATCH_SIZE = 32
N_EPOCHS = 5
LEARNING_RATE = 1e-3
TEACHER_FORCING_START = 1.0
TEACHER_FORCING_END = 0.2
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.1
VAL_SPLIT = 0.1
EARLY_STOPPING_PATIENCE = 3
USE_ATTENTION = True
ENC_BIDIRECTIONAL = True

# Optional MLflow configuration via environment variables:
# - MLFLOW_TRACKING_URI: http://host:port of your MLflow server
# - MLFLOW_EXPERIMENT_NAME: name of the experiment to group runs
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "seq2seq_translation")


def _teacher_forcing_ratio(epoch_idx, n_epochs, start, end):
    if n_epochs <= 1:
        return float(end)
    progress = (epoch_idx - 1) / (n_epochs - 1)
    return float(start + (end - start) * progress)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    autocast_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=torch.float16)
        if (use_amp and device.type == "cuda")
        else nullcontext()
    )

    for batch in dataloader:
        src, trg, src_lengths, _ = batch
        src = src.to(device)
        trg = trg.to(device)

        with autocast_ctx:
            output = model(src, src_lengths, trg, teacher_forcing_ratio=0.0)
            output_dim = output.shape[-1]

            output_flat = output[1:].reshape(-1, output_dim)
            trg_flat = trg[1:].reshape(-1)
            loss = criterion(output_flat, trg_flat)

        total_loss += loss.item()

        pred = output.argmax(-1)[1:]  # (trg_len-1, batch)
        gold = trg[1:]
        mask = gold != PAD_token
        total_correct += ((pred == gold) & mask).sum().item()
        total_tokens += mask.sum().item()

    avg_loss = total_loss / max(1, len(dataloader))
    token_acc = total_correct / max(1, total_tokens)
    return avg_loss, token_acc


def train(
    run_name=None,
    n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    label_smoothing=LABEL_SMOOTHING,
    val_split=VAL_SPLIT,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    teacher_forcing_start=TEACHER_FORCING_START,
    teacher_forcing_end=TEACHER_FORCING_END,
    grad_clip=GRAD_CLIP,
    use_attention=USE_ATTENTION,
    enc_bidirectional=ENC_BIDIRECTIONAL,
    seed=42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    print(f"Using device: {device}")

    run_ctx = nullcontext()
    use_mlflow = mlflow is not None

    if use_mlflow:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        effective_run_name = run_name or f"seq2seq_train_{int(time.time())}"
        run_ctx = mlflow.start_run(run_name=effective_run_name)

    with run_ctx:
        if use_mlflow:
            mlflow.log_params(
                {
                    "data_path": DATA_PATH,
                    "embedding_dim": embedding_dim,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "learning_rate": learning_rate,
                    "teacher_forcing_start": teacher_forcing_start,
                    "teacher_forcing_end": teacher_forcing_end,
                    "grad_clip": grad_clip,
                    "label_smoothing": label_smoothing,
                    "val_split": val_split,
                    "early_stopping_patience": early_stopping_patience,
                    "use_attention": use_attention,
                    "enc_bidirectional": enc_bidirectional,
                    "seed": seed,
                    "device": str(device),
                }
            )

        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(
                f"Dataset not found at {DATA_PATH}. "
                f"Place an English-French CSV there or run download_data.py first."
            )

        print(f"Loading and preparing data from {DATA_PATH} ...")
        input_lang, output_lang, pairs = prepareData(DATA_PATH, limit=None)
        print(f"Number of sentence pairs: {len(pairs)}")
        print(f"Input vocab size: {input_lang.n_words}, Output vocab size: {output_lang.n_words}")

        if use_mlflow:
            mlflow.log_params(
                {
                    "input_vocab_size": input_lang.n_words,
                    "output_vocab_size": output_lang.n_words,
                    "num_sentence_pairs": len(pairs),
                }
            )

        if not (0.0 <= val_split < 1.0):
            raise ValueError("val_split must be in [0.0, 1.0).")

        dataset = TranslationDataset(pairs, input_lang, output_lang)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        split_gen = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_gen)

        use_cuda = device.type == "cuda"
        num_workers = min(4, os.cpu_count() or 0)
        pin_memory = use_cuda

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
        val_loader = None
        if val_size > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=num_workers > 0,
            )

        total_batches = len(train_loader)
        print(
            f"Batch size: {batch_size}, Train batches/epoch: {total_batches}, "
            f"Train samples: {train_size}, Val samples: {val_size}"
        )

        encoder = EncoderRNN(
            input_vocab_size=input_lang.n_words,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=enc_bidirectional if use_attention else False,
        ).to(device)

        if use_attention:
            enc_output_dim = hidden_size * (2 if enc_bidirectional else 1)
            decoder = AttnDecoderRNN(
                output_vocab_size=output_lang.n_words,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                enc_output_dim=enc_output_dim,
                num_layers=num_layers,
                dropout=dropout,
            ).to(device)
            model = Seq2SeqAttn(encoder, decoder, device, enc_bidirectional=enc_bidirectional).to(device)
        else:
            decoder = DecoderRNN(
                output_vocab_size=output_lang.n_words,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            ).to(device)
            model = Seq2Seq(encoder, decoder, device).to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=PAD_token, label_smoothing=label_smoothing)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = (
            optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
            if val_loader is not None
            else optim.lr_scheduler.StepLR(optimizer, step_size=max(1, n_epochs // 3), gamma=0.5)
        )

        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler(enabled=use_amp)
        log_interval = max(1, total_batches // 10)  # ~10 logs / epoch

        global_step = 0
        best_metric = float("inf")
        epochs_no_improve = 0

        os.makedirs(MODEL_DIR, exist_ok=True)
        best_checkpoint_path = os.path.join(MODEL_DIR, CHECKPOINT_NAME)
        last_checkpoint_path = os.path.join(MODEL_DIR, "seq2seq_en_fr_last.pt")

        for epoch in range(1, n_epochs + 1):
            model.train()
            epoch_loss = 0.0

            start_time = time.time()
            teacher_forcing_ratio = _teacher_forcing_ratio(
                epoch, n_epochs, teacher_forcing_start, teacher_forcing_end
            )
            print(f"\n===== Epoch {epoch}/{n_epochs} (teacher_forcing={teacher_forcing_ratio:.3f}) =====")

            autocast_ctx = (
                torch.amp.autocast(device_type=device.type, dtype=torch.float16)
                if use_amp
                else nullcontext()
            )

            for batch_idx, batch in enumerate(train_loader, start=1):
                src, trg, src_lengths, _ = batch
                src = src.to(device)
                trg = trg.to(device)

                optimizer.zero_grad(set_to_none=True)

                with autocast_ctx:
                    output = model(
                        src,
                        src_lengths,
                        trg,
                        teacher_forcing_ratio=teacher_forcing_ratio,
                    )

                    output_dim = output.shape[-1]
                    output = output[1:].reshape(-1, output_dim)
                    trg_flat = trg[1:].reshape(-1)
                    loss = criterion(output, trg_flat)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

                batch_loss = loss.item()
                epoch_loss += batch_loss
                global_step += 1

                if use_mlflow:
                    mlflow.log_metric("train_loss_batch", batch_loss, step=global_step)

                if batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == total_batches:
                    avg_so_far = epoch_loss / batch_idx
                    print(
                        f"Epoch {epoch}/{n_epochs} "
                        f"- Batch {batch_idx}/{total_batches} "
                        f"- batch_loss: {batch_loss:.4f} "
                        f"- avg_loss: {avg_so_far:.4f}"
                    )

            train_loss = epoch_loss / max(1, total_batches)
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{n_epochs} completed - train_loss: {train_loss:.4f} - time: {elapsed:.1f}s")

            if use_mlflow:
                mlflow.log_metric("train_loss_epoch", train_loss, step=epoch)
                mlflow.log_metric("epoch_time_sec", elapsed, step=epoch)
                mlflow.log_metric("teacher_forcing_ratio", teacher_forcing_ratio, step=epoch)

            val_loss = None
            val_token_acc = None
            if val_loader is not None:
                val_loss, val_token_acc = evaluate(
                    model, val_loader, criterion, device, use_amp=use_amp
                )
                print(f"Validation - loss: {val_loss:.4f} - token_acc: {val_token_acc:.4f}")
                if use_mlflow:
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                    mlflow.log_metric("val_token_acc", val_token_acc, step=epoch)

            if val_loader is not None:
                scheduler.step(val_loss)
                metric = val_loss
            else:
                scheduler.step()
                metric = train_loss

            improved = metric < best_metric
            if improved:
                best_metric = metric
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            checkpoint = {
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "input_lang": input_lang,
                "output_lang": output_lang,
                "config": {
                    "embedding_dim": embedding_dim,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "use_attention": use_attention,
                    "enc_bidirectional": enc_bidirectional if use_attention else False,
                },
                "train_state": {
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                },
            }

            torch.save(checkpoint, last_checkpoint_path)
            if improved:
                torch.save(checkpoint, best_checkpoint_path)
                print(f"Saved new best model to {best_checkpoint_path} (best_metric={best_metric:.4f})")

            if use_mlflow:
                mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch)

            if val_loader is not None and epochs_no_improve >= early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch} (no improvement for {epochs_no_improve} epochs)."
                )
                break

        print(f"\nLast checkpoint saved to {last_checkpoint_path}")
        if os.path.exists(best_checkpoint_path):
            print(f"Best checkpoint saved to {best_checkpoint_path}")

        if use_mlflow:
            if os.path.exists(best_checkpoint_path):
                mlflow.log_artifact(best_checkpoint_path, artifact_path="models")
            mlflow.log_artifact(last_checkpoint_path, artifact_path="models")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Seq2Seq model with optional MLflow logging.")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional MLflow run name. If not set, a timestamp-based name is used.",
    )
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--embedding-dim", type=int, default=EMBEDDING_DIM)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--tf-start", type=float, default=TEACHER_FORCING_START)
    parser.add_argument("--tf-end", type=float, default=TEACHER_FORCING_END)
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--no-bidir", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    cli_args = parser.parse_args()

    train(
        run_name=cli_args.run_name,
        n_epochs=cli_args.epochs,
        batch_size=cli_args.batch_size,
        learning_rate=cli_args.lr,
        embedding_dim=cli_args.embedding_dim,
        hidden_size=cli_args.hidden_size,
        num_layers=cli_args.num_layers,
        dropout=cli_args.dropout,
        label_smoothing=cli_args.label_smoothing,
        val_split=cli_args.val_split,
        early_stopping_patience=cli_args.patience,
        teacher_forcing_start=cli_args.tf_start,
        teacher_forcing_end=cli_args.tf_end,
        use_attention=not cli_args.no_attention,
        enc_bidirectional=not cli_args.no_bidir,
        seed=cli_args.seed,
    )
