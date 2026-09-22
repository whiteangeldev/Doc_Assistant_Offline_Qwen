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

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "data"


@st.cache_resource(show_spinner=False)
def load_runtime():
    vectors, meta, config = load_index(INDEX_DIR)
    model, device, _ = load_embedder(DEFAULT_MODEL)
    return vectors, meta, config, model, device


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


def run_search(query, top_k, vectors, meta, config, model):
    import numpy as np

    prompt_name = config.get("query_prompt_name") or QUERY_PROMPT_NAME
    query_vector = np.asarray(encode_query(model, query, prompt_name), dtype=np.float32)
    return rank_hits(query_vector, vectors, meta, top_k)


def main():
    st.set_page_config(page_title="Offline document search", layout="wide")
    st.title("Offline document search")
    st.caption("Local Qwen3-Embedding-0.6B. Results are source sentences, not generated answers.")

    search_tab, = st.tabs(["SEARCH"])
    with search_tab:
        try:
            with st.spinner("Loading local index and embedding model..."):
                vectors, meta, config, model, device = load_runtime()
                port = pdf_port()
        except (ValueError, OSError, RuntimeError) as exc:
            st.error(str(exc))
            st.info("Build the index first: `.venv/bin/python build_index.py`")
            return

        st.caption(f"{len(meta):,} sentences indexed · {device}")

        with st.form("search"):
            query = st.text_area("Query", height=80, placeholder="Ask in any supported language")
            top_k = st.slider("Results", min_value=5, max_value=10, value=8)
            submitted = st.form_submit_button("Search")

        if submitted:
            query = (query or "").strip()
            if not query:
                st.warning("Enter a query.")
            else:
                with st.spinner("Searching…"):
                    st.session_state.hits = run_search(
                        query, top_k, vectors, meta, config, model,
                    )

        hits = st.session_state.get("hits")
        if not hits:
            return

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


if __name__ == "__main__":
    main()
