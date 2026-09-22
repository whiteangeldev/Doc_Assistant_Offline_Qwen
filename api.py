"""Local FastAPI search service. Load the model once; Express calls POST /api/search."""

import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from offline_runtime import DEFAULT_MODEL, QUERY_PROMPT_NAME, load_embedder
from search import encode_query, load_index, rank_hits
from search_db import TextDbWatcher, load_db_index, rank_db_hits
from sync_index import SourceWatcher

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "data"

runtime = {"ready": False, "error": None}
_lock = threading.Lock()
_stop = threading.Event()


def _apply_index(payload):
    runtime["vectors"] = payload["vectors"]
    runtime["meta"] = payload["meta"]
    if payload.get("config"):
        runtime["config"] = payload["config"]


def _apply_db_index(payload):
    runtime["db_vectors"] = payload["vectors"]
    runtime["db_meta"] = payload["meta"]
    if payload.get("config"):
        runtime["db_config"] = payload["config"]
    runtime["db_error"] = None


def _watch_source():
    watcher = SourceWatcher()
    db_watcher = TextDbWatcher()
    interval = min(watcher.interval, db_watcher.interval)
    while not _stop.is_set():
        if runtime.get("ready"):
            try:
                changed, payload = watcher.poll(
                    runtime["model"],
                    runtime["device"],
                    DEFAULT_MODEL,
                )
                if payload is not None:
                    with _lock:
                        _apply_index(payload)
                    if changed:
                        print(
                            f"Source sync: added {payload['added'] or '[]'} "
                            f"removed {payload['removed'] or '[]'} "
                            f"({payload['sentence_count']} sentences)",
                            flush=True,
                        )
            except Exception as exc:
                print(f"Source sync failed: {exc}", flush=True)
            try:
                changed, payload = db_watcher.poll(
                    runtime["model"],
                    runtime["device"],
                    DEFAULT_MODEL,
                )
                if payload is not None:
                    with _lock:
                        _apply_db_index(payload)
                    if changed:
                        print(
                            f"MySQL sync: added {payload['added'] or '[]'} "
                            f"removed {payload['removed'] or '[]'} "
                            f"({payload['row_count']} rows)",
                            flush=True,
                        )
            except Exception as exc:
                with _lock:
                    runtime["db_error"] = str(exc)
                print(f"MySQL sync failed: {exc}", flush=True)
        _stop.wait(interval)


@asynccontextmanager
async def lifespan(app):
    _stop.clear()
    try:
        vectors, meta, config = load_index(INDEX_DIR)
        model, device, _ = load_embedder(DEFAULT_MODEL)
        db_vectors, db_meta, db_config = load_db_index(INDEX_DIR)
        runtime.update(
            {
                "ready": True,
                "error": None,
                "vectors": np.asarray(vectors, dtype=np.float32),
                "meta": meta,
                "config": config,
                "model": model,
                "device": device,
                "prompt_name": config.get("query_prompt_name") or QUERY_PROMPT_NAME,
                "db_vectors": np.asarray(db_vectors, dtype=np.float32),
                "db_meta": db_meta,
                "db_config": db_config,
                "db_error": None,
            }
        )
        threading.Thread(target=_watch_source, name="source-watch", daemon=True).start()
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        runtime.update({"ready": False, "error": str(exc)})
    yield
    _stop.set()
    runtime.clear()
    runtime.update({"ready": False, "error": None})


app = FastAPI(title="Offline document search", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(8, ge=5, le=10)


def _require_ready():
    if not runtime.get("ready"):
        detail = (
            runtime.get("error") or "Search backend is not ready. Run build_index.py."
        )
        raise HTTPException(status_code=503, detail=detail)


@app.get("/api/health")
def health():
    _require_ready()
    with _lock:
        sentences = len(runtime["meta"])
        db_rows = len(runtime.get("db_meta") or [])
        db_error = runtime.get("db_error")
        device = runtime["device"]
    return {
        "ok": True,
        "sentences": sentences,
        "db_rows": db_rows,
        "db_error": db_error,
        "device": device,
        "watching": True,
    }


@app.post("/api/search")
def search_api(body: SearchRequest):
    _require_ready()
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is empty.")
    with _lock:
        query_vector = np.asarray(
            encode_query(runtime["model"], query, runtime["prompt_name"]),
            dtype=np.float32,
        )
        hits = rank_hits(query_vector, runtime["vectors"], runtime["meta"], body.top_k)
        device = runtime["device"]
    return {
        "query": query,
        "top_k": body.top_k,
        "source": "pdf",
        "device": device,
        "hits": hits,
    }


@app.post("/api/search/db")
def search_db_api(body: SearchRequest):
    _require_ready()
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is empty.")
    with _lock:
        if runtime.get("db_error") and not runtime.get("db_meta"):
            raise HTTPException(status_code=503, detail=runtime["db_error"])
        query_vector = np.asarray(
            encode_query(runtime["model"], query, runtime["prompt_name"]),
            dtype=np.float32,
        )
        hits = rank_db_hits(
            query_vector, runtime["db_vectors"], runtime["db_meta"], body.top_k,
        )
        device = runtime["device"]
    return {
        "query": query,
        "top_k": body.top_k,
        "source": "mysql",
        "device": device,
        "hits": hits,
    }


def main():
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
