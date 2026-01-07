import os
import time
import tempfile
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data_preprocessing import (
    TranslationDataset,
    collate_fn,
    PAD_token,
    build_langs,
    load_pairs,
    load_parallel_pairs,
)
from src.models import EncoderRNN, DecoderRNN, Seq2Seq, AttnDecoderRNN, Seq2SeqAttn

try:
    import mlflow
except ImportError:
    mlflow = None


def _looks_like_local_path(value: str) -> bool:
    value = value.strip()
    if not value:
        return False
    if value.startswith("file:"):
        return True
    if value.startswith("/"):
        return True
    if len(value) >= 3 and value[1:3] == ":\\":  # Windows drive path
        return True
    return False


def _is_not_found_error(exc: Exception) -> bool:
    text = str(exc)
    return "404" in text and "not found" in text.lower()


DATA_PATH = "data/tatoeba/fra.txt"
MODEL_DIR = "models"
CHECKPOINT_NAME = "seq2seq_en_fr.pt"

EMBEDDING_DIM = 256
HIDDEN_SIZE = 256
NUM_LAYERS = 1
DROPOUT = 0.1

BATCH_SIZE = 32
N_EPOCHS = 10
LEARNING_RATE = 1e-3
TEACHER_FORCING_START = 1.0
TEACHER_FORCING_END = 0.2
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.1
MAX_LENGTH = 15
INPUT_VOCAB_SIZE = 15000
OUTPUT_VOCAB_SIZE = 20000
MIN_WORD_FREQ = 2
TEXT_NORMALIZATION = "v2"
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


def _split_pairs(pairs, val_split, seed):
    if not pairs or val_split <= 0:
        return pairs, []
    if not (0.0 <= val_split < 1.0):
        raise ValueError("val_split must be in [0.0, 1.0).")
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(pairs), generator=g).tolist()
    val_size = int(len(pairs) * val_split)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]
    return train_pairs, val_pairs


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
    data_path=DATA_PATH,
    train_src=None,
    train_tgt=None,
    val_src=None,
    val_tgt=None,
    limit_pairs=None,
    n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    label_smoothing=LABEL_SMOOTHING,
    max_length=MAX_LENGTH,
    input_vocab_size=INPUT_VOCAB_SIZE,
    output_vocab_size=OUTPUT_VOCAB_SIZE,
    min_word_freq=MIN_WORD_FREQ,
    text_normalization=TEXT_NORMALIZATION,
    val_split=VAL_SPLIT,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    teacher_forcing_start=TEACHER_FORCING_START,
    teacher_forcing_end=TEACHER_FORCING_END,
    grad_clip=GRAD_CLIP,
    use_attention=USE_ATTENTION,
    enc_bidirectional=ENC_BIDIRECTIONAL,
    seed=42,
    skip_training=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    print(f"Using device: {device}")

    run_ctx = nullcontext()
    use_mlflow = mlflow is not None
    experiment_artifact_location = None

    if use_mlflow:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        try:
            experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        except Exception as exc:
            message = str(exc).lower()
            if "deleted experiment" in message:
                raise RuntimeError(
                    f"MLflow experiment '{MLFLOW_EXPERIMENT_NAME}' is deleted and cannot be set active. "
                    "Pick a new MLFLOW_EXPERIMENT_NAME or restore the experiment in the MLflow UI."
                ) from exc
            raise

        experiment_artifact_location = getattr(experiment, "artifact_location", None)
        if experiment_artifact_location and _looks_like_local_path(experiment_artifact_location):
            is_remote_tracking = bool(tracking_uri) and not _looks_like_local_path(tracking_uri)
            message = (
                "This MLflow experiment uses a local filesystem artifact location.\n"
                f"Tracking URI: {mlflow.get_tracking_uri()}\n"
                f"Experiment artifact location: {experiment_artifact_location}\n"
                "For remote tracking servers, experiments must use an artifact URI that the server can handle "
                "(typically `mlflow-artifacts:/...` when using `mlflow server --serve-artifacts ...`). "
                "Existing experiments keep their original artifact location, so after fixing the server create a NEW "
                "MLFLOW_EXPERIMENT_NAME."
            )
            if is_remote_tracking:
                raise RuntimeError(message)
            print(f"WARNING: {message}")

        effective_run_name = run_name or f"seq2seq_train_{int(time.time())}"
        run_ctx = mlflow.start_run(run_name=effective_run_name)

    with run_ctx:
        if use_mlflow:
            mlflow.log_params(
                {
                    "data_path": data_path,
                    "train_src": train_src or "",
                    "train_tgt": train_tgt or "",
                    "val_src": val_src or "",
                    "val_tgt": val_tgt or "",
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
                    "max_length": max_length,
                    "input_vocab_size_limit": input_vocab_size,
                    "output_vocab_size_limit": output_vocab_size,
                    "min_word_freq": min_word_freq,
                    "text_normalization": text_normalization,
                    "val_split": val_split,
                    "early_stopping_patience": early_stopping_patience,
                    "use_attention": use_attention,
                    "enc_bidirectional": enc_bidirectional,
                    "seed": seed,
                    "skip_training": skip_training,
                    "device": str(device),
                }
            )

        os.makedirs(MODEL_DIR, exist_ok=True)
        best_checkpoint_path = os.path.join(MODEL_DIR, CHECKPOINT_NAME)
        last_checkpoint_path = os.path.join(MODEL_DIR, "seq2seq_en_fr_last.pt")

        if skip_training:
            dummy_checkpoint = {
                "dummy": True,
                "created_at": int(time.time()),
                "note": "Artifact upload smoke-check from train.py (--skip-training).",
            }
            torch.save(dummy_checkpoint, best_checkpoint_path)
            torch.save(dummy_checkpoint, last_checkpoint_path)
            print("Skip-training mode: created dummy checkpoints:")
            print(f"- {best_checkpoint_path}")
            print(f"- {last_checkpoint_path}")

            if use_mlflow:
                active = mlflow.active_run()
                run_id = getattr(getattr(active, "info", None), "run_id", None)
                run_artifact_uri = getattr(getattr(active, "info", None), "artifact_uri", None)

                for local_path in (best_checkpoint_path, last_checkpoint_path):
                    try:
                        mlflow.log_artifact(local_path, artifact_path="models")
                    except Exception as exc:
                        print("ERROR: Failed to log artifact to MLflow.")
                        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
                        if experiment_artifact_location:
                            print(f"Experiment artifact location: {experiment_artifact_location}")
                        if run_artifact_uri:
                            print(f"Run artifact URI: {run_artifact_uri}")
                        print(f"Local artifact path: {local_path}")
                        raise

                try:
                    from mlflow.tracking import MlflowClient
                except Exception:
                    MlflowClient = None

                if MlflowClient is not None and run_id is not None:
                    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
                    for rel_path in (
                        f"models/{os.path.basename(best_checkpoint_path)}",
                        f"models/{os.path.basename(last_checkpoint_path)}",
                    ):
                        with tempfile.TemporaryDirectory() as tmpdir:
                            client.download_artifacts(run_id, rel_path, dst_path=tmpdir)

                print("OK: MLflow received model artifacts in skip-training mode.")
            return

        if (train_src is None) != (train_tgt is None):
            raise ValueError("Provide both train_src and train_tgt (or neither).")
        if (val_src is None) != (val_tgt is None):
            raise ValueError("Provide both val_src and val_tgt (or neither).")

        vocab_limit_src = None if (not input_vocab_size or input_vocab_size <= 0) else int(input_vocab_size)
        vocab_limit_tgt = None if (not output_vocab_size or output_vocab_size <= 0) else int(output_vocab_size)
        min_word_freq = max(1, int(min_word_freq))
        normalization = str(text_normalization)

        if train_src and train_tgt:
            if not os.path.exists(train_src):
                raise FileNotFoundError(f"Training source file not found: {train_src}")
            if not os.path.exists(train_tgt):
                raise FileNotFoundError(f"Training target file not found: {train_tgt}")

            print(f"Loading training data from {train_src} / {train_tgt} ...")
            train_pairs = load_parallel_pairs(
                train_src,
                train_tgt,
                limit=limit_pairs,
                max_length=max_length,
                normalization=normalization,
            )

            if val_src and val_tgt:
                if not os.path.exists(val_src):
                    raise FileNotFoundError(f"Validation source file not found: {val_src}")
                if not os.path.exists(val_tgt):
                    raise FileNotFoundError(f"Validation target file not found: {val_tgt}")
                print(f"Loading validation data from {val_src} / {val_tgt} ...")
                val_pairs = load_parallel_pairs(
                    val_src,
                    val_tgt,
                    limit=limit_pairs,
                    max_length=max_length,
                    normalization=normalization,
                )
            else:
                train_pairs, val_pairs = _split_pairs(train_pairs, val_split=val_split, seed=seed)
        else:
            if not os.path.exists(data_path):
                raise FileNotFoundError(
                    f"Dataset not found at {data_path}. "
                    "Run download_data.py first to download the Tatoeba (ManyThings) English-French dataset."
                )
            print(f"Loading and preparing data from {data_path} ...")
            all_pairs = load_pairs(
                data_path,
                limit=limit_pairs,
                max_length=max_length,
                normalization=normalization,
            )
            train_pairs, val_pairs = _split_pairs(all_pairs, val_split=val_split, seed=seed)

        if not train_pairs:
            raise RuntimeError(
                "No training pairs after filtering. Try increasing --max-length, switching --norm, or removing --limit."
            )

        input_lang, output_lang = build_langs(
            train_pairs,
            input_vocab_size=vocab_limit_src,
            output_vocab_size=vocab_limit_tgt,
            min_word_freq=min_word_freq,
            input_name="eng",
            output_name="fra",
        )

        print(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")
        print(f"Input vocab size: {input_lang.n_words}, Output vocab size: {output_lang.n_words}")

        if use_mlflow:
            mlflow.log_params(
                {
                    "train_pairs": len(train_pairs),
                    "val_pairs": len(val_pairs),
                    "input_vocab_size": input_lang.n_words,
                    "output_vocab_size": output_lang.n_words,
                }
            )

        train_dataset = TranslationDataset(train_pairs, input_lang, output_lang)
        train_size = len(train_dataset)
        val_dataset = None
        val_size = 0
        if val_pairs:
            val_dataset = TranslationDataset(val_pairs, input_lang, output_lang)
            val_size = len(val_dataset)

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
        if val_dataset is not None:
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
                    "max_length": max_length,
                    "input_vocab_size": input_vocab_size,
                    "output_vocab_size": output_vocab_size,
                    "min_word_freq": min_word_freq,
                    "text_normalization": text_normalization,
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
            active = mlflow.active_run()
            run_id = getattr(getattr(active, "info", None), "run_id", None)
            run_artifact_uri = getattr(getattr(active, "info", None), "artifact_uri", None)

            to_log = []
            if os.path.exists(best_checkpoint_path):
                to_log.append(best_checkpoint_path)
            to_log.append(last_checkpoint_path)

            for local_path in to_log:
                try:
                    mlflow.log_artifact(local_path, artifact_path="models")
                except PermissionError as exc:
                    print("ERROR: Failed to write artifact due to a permissions error.")
                    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
                    if experiment_artifact_location:
                        print(f"Experiment artifact location: {experiment_artifact_location}")
                    if run_artifact_uri:
                        print(f"Run artifact URI: {run_artifact_uri}")
                    print(f"Local artifact path: {local_path}")
                    print(f"Original error: {exc}")
                    print(
                        "Hint: Your MLflow server likely created the experiment with a local filesystem artifact root "
                        "(e.g. 'file:/mlruns'). In that mode, the *client* writes artifacts directly to that path. "
                        "For a remote MLflow server, configure an artifact store the client can access (S3/MinIO/NFS), "
                        "or start the server with artifact serving (MLflow: `mlflow server --serve-artifacts ...`)."
                    )
                    raise
                except Exception as exc:
                    print("ERROR: Failed to log artifact to MLflow.")
                    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
                    if experiment_artifact_location:
                        print(f"Experiment artifact location: {experiment_artifact_location}")
                    if run_artifact_uri:
                        print(f"Run artifact URI: {run_artifact_uri}")
                    print(f"Local artifact path: {local_path}")
                    print(f"Original error: {exc}")
                    raise

            try:
                from mlflow.tracking import MlflowClient
            except Exception:
                MlflowClient = None

            if MlflowClient is not None and run_id is not None:
                client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
                run_artifact_uri = run_artifact_uri or getattr(getattr(mlflow.active_run(), "info", None), "artifact_uri", None)

                expected = []
                if os.path.exists(best_checkpoint_path):
                    expected.append(f"models/{os.path.basename(best_checkpoint_path)}")
                expected.append(f"models/{os.path.basename(last_checkpoint_path)}")

                for rel_path in expected:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        try:
                            client.download_artifacts(run_id, rel_path, dst_path=tmpdir)
                        except Exception as exc:
                            print("ERROR: Model artifact upload may have failed (could not download it back).")
                            print(f"Run ID: {run_id}")
                            print(f"Expected artifact path: {rel_path}")
                            print(f"Run artifact URI: {run_artifact_uri}")
                            if experiment_artifact_location:
                                print(f"Experiment artifact location: {experiment_artifact_location}")
                            print(f"Original error: {exc}")
                            print(
                                "Hint: Run `python3 test_mlflow_artifact.py` to diagnose your MLflow artifact "
                                "configuration (remote servers often need `--serve-artifacts` or a shared artifact store)."
                            )
                            break

                try:
                    client.list_artifacts(run_id, path="models")
                except Exception as exc:
                    if not _is_not_found_error(exc):
                        print("WARNING: Could not list artifacts (non-fatal).")
                        print(f"Original error: {exc}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Seq2Seq model with optional MLflow logging.")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional MLflow run name. If not set, a timestamp-based name is used.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DATA_PATH,
        help="TSV/CSV dataset path (ignored if --train-src/--train-tgt are provided).",
    )
    parser.add_argument("--train-src", type=str, default=None, help="Training source file (one sentence per line).")
    parser.add_argument("--train-tgt", type=str, default=None, help="Training target file (one sentence per line).")
    parser.add_argument("--val-src", type=str, default=None, help="Validation source file (one sentence per line).")
    parser.add_argument("--val-tgt", type=str, default=None, help="Validation target file (one sentence per line).")
    parser.add_argument(
        "--limit",
        "--limit-pairs",
        dest="limit_pairs",
        type=int,
        default=None,
        help="Optional cap on number of pairs loaded (debug).",
    )
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--embedding-dim", type=int, default=EMBEDDING_DIM)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--input-vocab-size", type=int, default=INPUT_VOCAB_SIZE)
    parser.add_argument("--output-vocab-size", type=int, default=OUTPUT_VOCAB_SIZE)
    parser.add_argument("--min-word-freq", type=int, default=MIN_WORD_FREQ)
    parser.add_argument("--norm", type=str, default=TEXT_NORMALIZATION, choices=["v1", "v2"])
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--tf-start", type=float, default=TEACHER_FORCING_START)
    parser.add_argument("--tf-end", type=float, default=TEACHER_FORCING_END)
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--no-bidir", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip the training loop and only create/log dummy .pt artifacts to validate MLflow artifact upload.",
    )
    cli_args = parser.parse_args()

    train(
        run_name=cli_args.run_name,
        data_path=cli_args.data,
        train_src=cli_args.train_src,
        train_tgt=cli_args.train_tgt,
        val_src=cli_args.val_src,
        val_tgt=cli_args.val_tgt,
        limit_pairs=cli_args.limit_pairs,
        n_epochs=cli_args.epochs,
        batch_size=cli_args.batch_size,
        learning_rate=cli_args.lr,
        embedding_dim=cli_args.embedding_dim,
        hidden_size=cli_args.hidden_size,
        num_layers=cli_args.num_layers,
        dropout=cli_args.dropout,
        label_smoothing=cli_args.label_smoothing,
        max_length=cli_args.max_length,
        input_vocab_size=cli_args.input_vocab_size,
        output_vocab_size=cli_args.output_vocab_size,
        min_word_freq=cli_args.min_word_freq,
        text_normalization=cli_args.norm,
        val_split=cli_args.val_split,
        early_stopping_patience=cli_args.patience,
        teacher_forcing_start=cli_args.tf_start,
        teacher_forcing_end=cli_args.tf_end,
        use_attention=not cli_args.no_attention,
        enc_bidirectional=not cli_args.no_bidir,
        seed=cli_args.seed,
        skip_training=cli_args.skip_training,
    )
