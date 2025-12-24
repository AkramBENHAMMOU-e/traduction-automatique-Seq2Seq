import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data_preprocessing import (
    prepareData,
    TranslationDataset,
    collate_fn,
    PAD_token,
)
from src.models import EncoderRNN, DecoderRNN, Seq2Seq

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
TEACHER_FORCING_RATIO = 0.5
GRAD_CLIP = 1.0

# Optional MLflow configuration via environment variables:
# - MLFLOW_TRACKING_URI: http://host:port of your MLflow server
# - MLFLOW_EXPERIMENT_NAME: name of the experiment to group runs
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "seq2seq_translation")


def train(run_name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

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
                    "embedding_dim": EMBEDDING_DIM,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "batch_size": BATCH_SIZE,
                    "n_epochs": N_EPOCHS,
                    "learning_rate": LEARNING_RATE,
                    "teacher_forcing_ratio": TEACHER_FORCING_RATIO,
                    "grad_clip": GRAD_CLIP,
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

        dataset = TranslationDataset(pairs, input_lang, output_lang)
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
        )

        total_batches = len(dataloader)
        print(f"Batch size: {BATCH_SIZE}, Total batches per epoch: {total_batches}")

        encoder = EncoderRNN(
            input_vocab_size=input_lang.n_words,
            embedding_dim=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ).to(device)

        decoder = DecoderRNN(
            output_vocab_size=output_lang.n_words,
            embedding_dim=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ).to(device)

        model = Seq2Seq(encoder, decoder, device).to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        log_interval = max(1, total_batches // 10)  # ~10 logs / epoch

        global_step = 0

        for epoch in range(1, N_EPOCHS + 1):
            model.train()
            epoch_loss = 0.0

            start_time = time.time()
            print(f"\n===== Epoch {epoch}/{N_EPOCHS} =====")

            for batch_idx, batch in enumerate(dataloader, start=1):
                src, trg, src_lengths, _ = batch

                src = src.to(device)
                trg = trg.to(device)

                optimizer.zero_grad()

                output = model(
                    src,
                    src_lengths,
                    trg,
                    teacher_forcing_ratio=TEACHER_FORCING_RATIO,
                )

                # output: (trg_len, batch_size, vocab_size)
                output_dim = output.shape[-1]

                output = output[1:].reshape(-1, output_dim)
                trg_flat = trg[1:].reshape(-1)

                loss = criterion(output, trg_flat)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss
                global_step += 1

                if use_mlflow:
                    mlflow.log_metric("train_loss_batch", batch_loss, step=global_step)

                if (
                    batch_idx == 1
                    or batch_idx % log_interval == 0
                    or batch_idx == total_batches
                ):
                    avg_so_far = epoch_loss / batch_idx
                    print(
                        f"Epoch {epoch}/{N_EPOCHS} "
                        f"- Batch {batch_idx}/{total_batches} "
                        f"- batch_loss: {batch_loss:.4f} "
                        f"- avg_loss: {avg_so_far:.4f}"
                    )

            avg_loss = epoch_loss / total_batches
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch}/{N_EPOCHS} completed - avg_loss: {avg_loss:.4f} - time: {elapsed:.1f}s"
            )

            if use_mlflow:
                mlflow.log_metric("train_loss_epoch", avg_loss, step=epoch)
                mlflow.log_metric("epoch_time_sec", elapsed, step=epoch)

        os.makedirs(MODEL_DIR, exist_ok=True)
        checkpoint_path = os.path.join(MODEL_DIR, CHECKPOINT_NAME)

        torch.save(
            {
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "input_lang": input_lang,
                "output_lang": output_lang,
                "config": {
                    "embedding_dim": EMBEDDING_DIM,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                },
            },
            checkpoint_path,
        )

        print(f"\nTraining finished. Model saved to {checkpoint_path}")

        if use_mlflow:
            mlflow.log_artifact(checkpoint_path, artifact_path="models")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Seq2Seq model with optional MLflow logging.")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional MLflow run name. If not set, a timestamp-based name is used.",
    )
    cli_args = parser.parse_args()

    train(run_name=cli_args.run_name)
