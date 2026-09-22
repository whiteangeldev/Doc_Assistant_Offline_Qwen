"""Step 5: local Streamlit SEARCH over the sentence index."""

import os
import subprocess
import sys
import webbrowser
from pathlib import Path

os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

import streamlit as st

from offline_runtime import DEFAULT_MODEL, QUERY_PROMPT_NAME, load_embedder
from highlight_pdf import highlight_hit
from pdf_server import SOURCE_DIR, highlight_url, start_pdf_server
from search import encode_query, load_index, rank_hits
from search_db import TextDbWatcher, load_db_index, rank_db_hits
from sync_index import SOURCE_DIR as WATCH_SOURCE
from sync_index import SourceWatcher

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "data"


@st.cache_resource(show_spinner=False)
def load_runtime():
    vectors, meta, config = load_index(INDEX_DIR)
    model, device, _ = load_embedder(DEFAULT_MODEL)
    import numpy as np

    db_vectors, db_meta, db_config = load_db_index(INDEX_DIR)
    return {
        "vectors": np.asarray(vectors, dtype=np.float32),
        "meta": meta,
        "config": config,
        "model": model,
        "device": device,
        "watcher": SourceWatcher(WATCH_SOURCE, INDEX_DIR),
        "db_vectors": np.asarray(db_vectors, dtype=np.float32),
        "db_meta": db_meta,
        "db_config": db_config,
        "db_watcher": TextDbWatcher(INDEX_DIR),
    }


@st.cache_resource(show_spinner=False)
def pdf_port():
    # Versioned so a code reload starts a server that can serve /hl/ copies.
    return start_pdf_server(SOURCE_DIR)


def open_pdf_at_page(hit, port):
    """Open a highlighted copy of the PDF in the browser at the cited page."""
    source = (SOURCE_DIR / hit["file"]).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"PDF not found: {source}")
    highlighted, marks = highlight_hit(source, hit["page"], hit["text"], hit["id"])
    url = highlight_url(port, highlighted.name, hit["page"])
    if sys.platform == "darwin":
        subprocess.run(["open", url], check=True)
    else:
        webbrowser.open(url)
    return url, marks


def _rank_query(query, top_k, vectors, meta, config, model, ranker):
    import numpy as np

    prompt_name = (config or {}).get("query_prompt_name") or QUERY_PROMPT_NAME
    query_vector = np.asarray(encode_query(model, query, prompt_name), dtype=np.float32)
    return ranker(query_vector, vectors, meta, top_k)


def run_search(query, top_k, vectors, meta, config, model):
    return _rank_query(query, top_k, vectors, meta, config, model, rank_hits)


def run_db_search(query, top_k, vectors, meta, config, model):
    return _rank_query(query, top_k, vectors, meta, config, model, rank_db_hits)


def _apply_pdf_payload(state, payload, changed):
    state["vectors"] = payload["vectors"]
    state["meta"] = payload["meta"]
    if payload.get("config"):
        state["config"] = payload["config"]
    if changed:
        added = ", ".join(payload["added"]) or "none"
        removed = ", ".join(payload["removed"]) or "none"
        st.session_state.source_sync_msg = (
            f"Source updated. Added {added}; removed {removed}; "
            f"{payload['sentence_count']} searchable sentences."
        )


def _apply_db_payload(state, payload, changed):
    state["db_vectors"] = payload["vectors"]
    state["db_meta"] = payload["meta"]
    if payload.get("config"):
        state["db_config"] = payload["config"]
    if changed:
        added = ", ".join(str(item) for item in payload["added"]) or "none"
        removed = ", ".join(str(item) for item in payload["removed"]) or "none"
        st.session_state.db_sync_msg = (
            f"MySQL updated. Added {added}; removed {removed}; "
            f"{payload['row_count']} searchable rows."
        )


def main():
    st.set_page_config(page_title="Offline document search", layout="wide")
    st.title("Offline document search")
    st.caption("Local Qwen3-Embedding-0.6B. Results are retrieved text, not generated answers.")

    try:
        with st.spinner("Loading local index and embedding model..."):
            state = load_runtime()
            port = pdf_port()
    except (ValueError, OSError, RuntimeError) as exc:
        st.error(str(exc))
        st.info("Build the PDF index first: `.venv/bin/python build_index.py`")
        return

    pdf_tab, text_tab = st.tabs(["PDF", "MySQL text"])

    with pdf_tab:
        try:
            changed, payload = state["watcher"].poll(
                state["model"], state["device"], DEFAULT_MODEL,
            )
            if payload is not None:
                _apply_pdf_payload(state, payload, changed)
        except Exception as exc:
            st.warning(f"Source sync failed: {exc}")

        if st.session_state.get("source_sync_msg"):
            st.success(st.session_state.pop("source_sync_msg"))

        st.caption(f"{len(state['meta']):,} PDF sentences indexed · {state['device']}")

        @st.fragment(run_every=2)
        def _watch_source():
            try:
                changed, payload = state["watcher"].poll(
                    state["model"], state["device"], DEFAULT_MODEL,
                )
            except Exception as exc:
                st.warning(f"Source sync failed: {exc}")
                return
            if payload is None:
                return
            _apply_pdf_payload(state, payload, changed)
            st.rerun()

        _watch_source()

        with st.form("search"):
            query = st.text_area("Query", height=80, placeholder="Ask in any supported language")
            top_k = st.slider("Results", min_value=5, max_value=10, value=8)
            submitted = st.form_submit_button("Search PDFs")

        if submitted:
            query = (query or "").strip()
            if not query:
                st.warning("Enter a query.")
            else:
                with st.spinner("Searching…"):
                    st.session_state.hits = run_search(
                        query, top_k, state["vectors"], state["meta"],
                        state["config"], state["model"],
                    )

        hits = st.session_state.get("hits")
        if hits:
            st.subheader(f"Top {len(hits)} hits")
            st.caption("Score is ranking similarity, not factual confidence.")
            for hit in hits:
                location = f"{hit['file']} page {hit['page']}"
                with st.container(border=True):
                    st.markdown(f"**{hit['score']:.3f}** · {location}")
                    st.write(hit["text"])
                    if st.button("Open PDF at page", key=f"open-{hit['id']}"):
                        try:
                            url, marks = open_pdf_at_page(hit, port)
                            if marks:
                                st.success(
                                    f"Opened {hit['file']} page {hit['page']} "
                                    f"with the matching sentence highlighted."
                                )
                            else:
                                st.warning(
                                    f"Opened {hit['file']} page {hit['page']}, "
                                    "but the exact sentence could not be located on the page."
                                )
                            st.caption(url)
                        except OSError as exc:
                            st.error(str(exc))

    with text_tab:
        try:
            changed, payload = state["db_watcher"].poll(
                state["model"], state["device"], DEFAULT_MODEL,
            )
            if payload is not None:
                _apply_db_payload(state, payload, changed)
        except Exception as exc:
            st.warning(f"MySQL sync failed: {exc}")
            st.info("Create the local database with `.venv/bin/python text_db.py --init`")

        if st.session_state.get("db_sync_msg"):
            st.success(st.session_state.pop("db_sync_msg"))

        st.caption(f"{len(state['db_meta']):,} MySQL rows indexed · {state['device']}")

        @st.fragment(run_every=2)
        def _watch_db():
            try:
                changed, payload = state["db_watcher"].poll(
                    state["model"], state["device"], DEFAULT_MODEL,
                )
            except Exception as exc:
                st.warning(f"MySQL sync failed: {exc}")
                return
            if payload is None:
                return
            _apply_db_payload(state, payload, changed)
            st.rerun()

        _watch_db()

        with st.form("search-db"):
            query = st.text_area(
                "Query",
                height=80,
                placeholder="Search title and content rows",
            )
            top_k = st.slider("Results", min_value=5, max_value=10, value=8, key="db_top_k")
            submitted = st.form_submit_button("Search MySQL")

        if submitted:
            query = (query or "").strip()
            if not query:
                st.warning("Enter a query.")
            else:
                with st.spinner("Searching…"):
                    st.session_state.db_hits = run_db_search(
                        query, top_k, state["db_vectors"], state["db_meta"],
                        state["db_config"], state["model"],
                    )

        hits = st.session_state.get("db_hits")
        if hits:
            st.subheader(f"Top {len(hits)} hits")
            st.caption("Each hit is one MySQL row. Score is ranking similarity.")
            for hit in hits:
                with st.container(border=True):
                    st.markdown(f"**{hit['score']:.3f}** · #{hit['id']} · {hit['title']}")
                    st.write(hit["content"])


if __name__ == "__main__":
    main()
