## 1. Clone the repository

```bash
git clone https://github.com/AkramBENHAMMOU-e/traduction-automatique-Seq2Seq
cd traduction-automatique-Seq2Seq
```

## 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs PyTorch, pandas, MLflow, etc.

## 4. Download the dataset (Tatoeba / ManyThings)

This project uses the English–French sentence pairs published via ManyThings (Tatoeba).
Run:

```bash
python download_data.py
```

The dataset will be extracted to `data/tatoeba/fra.txt`.

### Optional (recommended): create fixed train/val/test files

This mirrors the `seq2seq-attn-ref` style dataset structure (one sentence per line, separate files per split).

```bash
python prepare_splits.py --data data/tatoeba/fra.txt --out-dir data/splits --norm v2 --max-length 15 --seed 42 --val-split 0.1 --test-split 0.1
```

This writes:

- `data/splits/train.en`, `data/splits/train.fr`
- `data/splits/val.en`, `data/splits/val.fr`
- `data/splits/test.en`, `data/splits/test.fr`

## 5. Train the Seq2Seq model

Basic training (CPU or GPU if available):

```bash
python train.py
```

This will:

- Load `data/tatoeba/fra.txt`.
- Train an encoder–decoder Seq2Seq model (by default: bidirectional encoder + attention).
- Split data into train/validation and save the best checkpoint.
- Save the checkpoint to `models/seq2seq_en_fr.pt`.

Useful flags:

```bash
python train.py --epochs 20 --batch-size 64 --lr 3e-4
python train.py --no-attention  # baseline without attention
python train.py --val-split 0.1 --patience 3 --tf-start 1.0 --tf-end 0.2
python train.py --input-vocab-size 15000 --output-vocab-size 20000 --min-word-freq 2 --norm v2
python train.py --train-src data/splits/train.en --train-tgt data/splits/train.fr --val-src data/splits/val.en --val-tgt data/splits/val.fr
python train.py --skip-training  # quick MLflow artifact upload check
```

You can optionally name the MLflow run:

```bash
python train.py --run-name "exp_baseline_lr1e-3_bs32"
```

## 6. Run inference (translation)

Once the model is trained and the checkpoint exists in `models/seq2seq_en_fr.pt`:

```bash
python translate.py
```

Or use a specific checkpoint file:

```bash
python translate.py --checkpoint models/seq2seq_en_fr_last.pt
```

Then type an English sentence, for example:

```text
> i am happy today .
French: je suis heureux aujourd hui .
```

Press Enter on an empty line to quit.

Beam search options (usually improves output quality):

```bash
python translate.py --beam-size 5 --length-penalty 0.6 --max-length 30
```

## 7. MLflow tracking (optional)

This project integrates MLflow to track experiments (hyperparameters, metrics, and model artifacts).

1. Start an MLflow tracking server (example):

   ```bash
   mlflow server \
     --backend-store-uri sqlite:////path/to/mlflow.db \
     --default-artifact-root file:/path/to/mlflow_artifacts \
     --host 0.0.0.0 \
     --port 5000
   ```

2. Configure the tracking URI (local shell):

   ```bash
   export MLFLOW_TRACKING_URI="http://<host>:5000"
   # optionally
   export MLFLOW_EXPERIMENT_NAME="seq2seq_translation"
   ```

3. Test the connection:

   ```bash
   python test_mlflow_connection.py
   ```

   You should see a message confirming that a test run was created.

4. Run training with MLflow logging enabled:

   ```bash
   python train.py --run-name "exp_with_mlflow"
   ```

Open the MLflow UI in your browser at `http://<host>:5000` to inspect runs, metrics, and artifacts.

## 9. Compare two checkpoints

1. Keep both checkpoints (rename the old one before retraining), e.g.:

```bash
mv models/seq2seq_en_fr.pt models/seq2seq_old.pt
```

2. Train the new model (it will write `models/seq2seq_en_fr.pt` again), then run:

```bash
python compare_models.py --ckpt-a models/seq2seq_old.pt --ckpt-b models/seq2seq_en_fr.pt --test-split 0.1 --seed 42
```

If you created fixed split files:

```bash
python compare_models.py --ckpt-a models/seq2seq_old.pt --ckpt-b models/seq2seq_en_fr.pt --test-src data/splits/test.en --test-tgt data/splits/test.fr --norm v2
```

## 8. Running in Google Colab

In Colab, you can:

```python
!git clone https://github.com/AkramBENHAMMOU-e/traduction-automatique-Seq2Seq
%cd traduction-automatique-Seq2Seq
!pip install -r requirements.txt
!python download_data.py

import os
os.environ["MLFLOW_TRACKING_URI"] = "http://<host>:5000"  # optional
os.environ["MLFLOW_EXPERIMENT_NAME"] = "seq2seq_translation"

!python train.py --run-name "colab_run"
```

Replace `<host>` with the public address or tunnel URL of your MLflow server.
