"""Smoke test: verify an MLflow tracking server stores artifacts.

Usage:
  export MLFLOW_TRACKING_URI="http://<host>:5000"
  python3 test_mlflow_artifact.py

Exit codes:
  0: artifact successfully logged and visible via MLflow API
  1: artifact not found / MLflow misconfigured
  2: mlflow not installed
"""

from __future__ import annotations

import os
import sys
import tempfile
import time


def _looks_like_local_path(value: str) -> bool:
    value = value.strip()
    if not value:
        return False
    if value.startswith("file:"):
        return True
    if value.startswith("/"):
        return True
    if len(value) >= 3 and value[1:3] == ":\\":
        return True
    return False


def _is_not_found_error(exc: Exception) -> bool:
    text = str(exc)
    return "404" in text and "not found" in text.lower()


def main() -> int:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception as exc:
        print(f"ERROR: mlflow is not available ({exc}).", file=sys.stderr)
        return 2

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print(
            "ERROR: MLFLOW_TRACKING_URI is not set. Example: export MLFLOW_TRACKING_URI=\"http://localhost:5000\"",
            file=sys.stderr,
        )
        return 1

    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "mlflow_artifact_smoke_test")
    try:
        experiment = mlflow.set_experiment(experiment_name)
    except Exception as exc:
        message = str(exc).lower()
        if "deleted experiment" in message:
            print(
                f"ERROR: Experiment '{experiment_name}' is deleted and cannot be set active.",
                file=sys.stderr,
            )
            print(
                "Fix: pick a new experiment name, e.g.: "
                "export MLFLOW_EXPERIMENT_NAME=mlflow_artifact_smoke_test_v2",
                file=sys.stderr,
            )
            print(
                "Alternative: restore the deleted experiment in the MLflow UI (Experiments page) "
                "or permanently delete it, then re-create it.",
                file=sys.stderr,
            )
            return 1
        raise
    experiment_artifact_location = getattr(experiment, "artifact_location", None)
    if experiment_artifact_location and _looks_like_local_path(experiment_artifact_location):
        print("ERROR: This experiment uses a local filesystem artifact location.", file=sys.stderr)
        print(f"Tracking URI: {tracking_uri}", file=sys.stderr)
        print(f"Experiment artifact location: {experiment_artifact_location}", file=sys.stderr)
        print(
            "Hint: For remote tracking servers, experiments must use an artifact URI that the server can handle "
            "(typically `mlflow-artifacts:/...` when using `mlflow server --serve-artifacts ...`). "
            "Existing experiments keep their original artifact location, so create a NEW experiment name "
            "after fixing the server (e.g. `export MLFLOW_EXPERIMENT_NAME=mlflow_artifact_smoke_test_v2`).",
            file=sys.stderr,
        )
        return 1

    artifact_dir = "artifact_smoke"
    artifact_name = f"artifact_{int(time.time())}.txt"
    expected_rel_path = f"{artifact_dir}/{artifact_name}"

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, artifact_name)
        with open(local_path, "w", encoding="utf-8") as f:
            expected_contents = "mlflow artifact smoke test\n"
            f.write(expected_contents)

        with mlflow.start_run(run_name=f"artifact_smoke_{int(time.time())}") as run:
            artifact_uri = run.info.artifact_uri

            try:
                mlflow.log_artifact(local_path, artifact_path=artifact_dir)
            except PermissionError as exc:
                print("ERROR: Failed to write artifact due to a permissions error.", file=sys.stderr)
                print(f"Tracking URI: {tracking_uri}", file=sys.stderr)
                if experiment_artifact_location:
                    print(f"Experiment artifact location: {experiment_artifact_location}", file=sys.stderr)
                print(f"Run artifact URI: {artifact_uri}", file=sys.stderr)
                print(f"Original error: {exc}", file=sys.stderr)
                print(
                    "Hint: Your MLflow server likely created the experiment with a local filesystem artifact root "
                    "(e.g. 'file:/mlruns'). In that mode, the *client* writes artifacts directly to that path. "
                    "For a remote MLflow server, configure an artifact store the client can access (S3/MinIO/NFS), "
                    "or start the server with artifact serving (MLflow: 'mlflow server --serve-artifacts ...').",
                    file=sys.stderr,
                )
                return 1
            except Exception as exc:
                print("ERROR: Failed to log artifact to MLflow.", file=sys.stderr)
                print(f"Tracking URI: {tracking_uri}", file=sys.stderr)
                if experiment_artifact_location:
                    print(f"Experiment artifact location: {experiment_artifact_location}", file=sys.stderr)
                print(f"Run artifact URI: {artifact_uri}", file=sys.stderr)
                print(f"Original error: {exc}", file=sys.stderr)
                return 1

            client = MlflowClient(tracking_uri=tracking_uri)
            with tempfile.TemporaryDirectory() as download_dir:
                try:
                    downloaded_path = client.download_artifacts(
                        run.info.run_id,
                        expected_rel_path,
                        dst_path=download_dir,
                    )
                except Exception as exc:
                    print("ERROR: Artifact upload may have failed (could not download it back).", file=sys.stderr)
                    print(f"Run ID: {run.info.run_id}", file=sys.stderr)
                    print(f"Expected artifact path: {expected_rel_path}", file=sys.stderr)
                    print(f"Run artifact URI: {artifact_uri}", file=sys.stderr)
                    print(f"Original error: {exc}", file=sys.stderr)
                    return 1

                try:
                    with open(downloaded_path, "r", encoding="utf-8") as f:
                        got = f.read()
                except Exception as exc:
                    print("ERROR: Downloaded artifact could not be read.", file=sys.stderr)
                    print(f"Downloaded path: {downloaded_path}", file=sys.stderr)
                    print(f"Original error: {exc}", file=sys.stderr)
                    return 1

                if got != expected_contents:
                    print("ERROR: Downloaded artifact content mismatch.", file=sys.stderr)
                    print(f"Downloaded path: {downloaded_path}", file=sys.stderr)
                    return 1

            # Best-effort listing: some servers/reverse-proxies may not implement all newer endpoints.
            try:
                client.list_artifacts(run.info.run_id, path=artifact_dir)
            except Exception as exc:
                if not _is_not_found_error(exc):
                    print("WARNING: Could not list artifacts (non-fatal).", file=sys.stderr)
                    print(f"Original error: {exc}", file=sys.stderr)

            print("OK: MLflow received artifact.")
            print(f"Run ID: {run.info.run_id}")
            print(f"Artifact: {expected_rel_path}")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
