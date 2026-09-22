"""Load the local Qwen embedding model with network access disabled."""

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "Qwen3-Embedding-0.6B"
QUERY_PROMPT_NAME = "query"
EMBEDDING_DIM = 1024
# Sentences are short; cap length so one table-dump cannot OOM a batch.
MAX_SEQ_LENGTH = 512


def pick_device(preferred=None):
    if preferred:
        return preferred
    import torch
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def load_embedder(model_path=None, device=None):
    from sentence_transformers import SentenceTransformer

    path = Path(model_path) if model_path else DEFAULT_MODEL
    if not path.is_dir() or not (path / "model.safetensors").is_file():
        raise FileNotFoundError(f"Local embedding model not found at {path}")
    device = pick_device(device)
    model = SentenceTransformer(str(path), local_files_only=True, device=device)
    return model, device, path.resolve()
