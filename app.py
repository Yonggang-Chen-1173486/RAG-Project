"""
Aventro Motors RAG Assistant — Streamlit Web UI.

Features:
  - Chat interface with conversation history
  - Pipeline selector (simple / flexible / advanced)
  - Retrieval controls (top_k, min_score)
  - Source viewer with similarity scores
  - Evaluation dashboard (summary, per-question data, CSV download)
  - Optional user API keys (falls back to server-side defaults)
  - Session-scoped document uploads (isolated per browser session)

Run with:
    streamlit run app.py
"""
import sys
import json
import uuid
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import os
os.environ["HF_HUB_VERBOSITY"] = "error"

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd

from rag.embedding.embedding_manager import EmbeddingManager
from rag.retrieval.vector_store import VectorStore
from rag.retrieval.retriever import RAGRetriever
from rag.generation.llm import GroqLLM
from rag.pipeline import rag_simple, rag_flexible, AdvancedRAGPipeline
from rag.ingestion.loaders import load_file
from rag.ingestion.splitter import split_documents


# ============================================================
# Paths
# ============================================================
_PROJECT_ROOT = Path(__file__).resolve().parent
_EVAL_RESULTS_DIR = _PROJECT_ROOT / "evaluation" / "results"
_EVAL_SUMMARY = _EVAL_RESULTS_DIR / "summary.json"
_EVAL_REPORT = _EVAL_RESULTS_DIR / "report.md"
_EVAL_CSV = _EVAL_RESULTS_DIR / "results.csv"


# ============================================================
# API key resolution
# ============================================================
def get_secret_or_env(name: str) -> str | None:
    """Read API key from Streamlit secrets, falling back to env var."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except (FileNotFoundError, KeyError):
        pass
    return os.environ.get(name)


# Populate environment variables from secrets (if not already set)
os.environ.setdefault("GROQ_API_KEY", get_secret_or_env("GROQ_API_KEY") or "")
os.environ.setdefault("GOOGLE_API_KEY", get_secret_or_env("GOOGLE_API_KEY") or "")


# ============================================================
# MultiRetriever — merges main store + user uploads
# ============================================================
class MultiRetriever:
    """Retrieves from multiple vector stores and merges results by score."""

    def __init__(self, retrievers: list):
        self.retrievers = retrievers

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.7):
        merged = []
        for r in self.retrievers:
            merged.extend(r.retrieve(query, top_k=top_k, score_threshold=score_threshold))
        merged.sort(key=lambda x: x["similarity_score"], reverse=True)
        return merged[:top_k]


# ============================================================
# Cached initialization (main index only — user index is per-session)
# ============================================================
def ensure_vector_store():
    """Build the vector store from source documents if the collection is empty.

    Runs silently on startup — no user-facing UI is shown.
    """
    from rag.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
    import chromadb

    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    col = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"description": "PDF document embeddings for RAG", "hnsw:space": "cosine"},
    )

    if col.count() > 0:
        return  # already populated

    # Empty → build it silently
    from rag.embedding.embedding_manager import EmbeddingManager
    from rag.retrieval.vector_store import VectorStore
    from rag.ingestion.loaders import process_all_documents
    from rag.ingestion.splitter import split_documents
    from rag.ingestion.indexer import index_documents

    em = EmbeddingManager()
    vs = VectorStore()
    docs = process_all_documents()
    chunks = split_documents(docs)
    index_documents(chunks, em, vs)


@st.cache_resource
def init_rag():
    ensure_vector_store()
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()
    retriever = RAGRetriever(vector_store, embedding_manager)
    llm = GroqLLM()
    return embedding_manager, retriever, llm


# ============================================================
# User upload handling (session-scoped)
# ============================================================
def build_user_collection(uploaded_files, session_id: str):
    """Parse, chunk, embed uploaded files into a session-scoped Chroma collection."""
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"uploads_{session_id}_"))

    docs = []
    for uf in uploaded_files:
        file_path = tmp_dir / uf.name
        file_path.write_bytes(uf.getbuffer())
        try:
            docs.extend(load_file(file_path, tmp_dir))
        except Exception as e:
            st.sidebar.warning(f"Could not load {uf.name}: {e}")

    if not docs:
        return None, 0

    chunks = split_documents(docs)

    collection_name = f"user_{session_id}"
    vector_store = VectorStore(collection_name=collection_name)
    embedding_manager = EmbeddingManager()
    texts = [c.page_content for c in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)
    vector_store.add_documents(chunks, embeddings)

    return vector_store, len(chunks)


# ============================================================
# Evaluation data loaders
# ============================================================
@st.cache_data
def load_evaluation_summary():
    if not _EVAL_SUMMARY.exists():
        return None
    with open(_EVAL_SUMMARY, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_evaluation_csv():
    if not _EVAL_CSV.exists():
        return None
    return pd.read_csv(_EVAL_CSV)


@st.cache_data
def load_evaluation_report():
    if not _EVAL_REPORT.exists():
        return None
    with open(_EVAL_REPORT, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="Aventro Motors RAG", page_icon="🚗", layout="wide")
st.title("🚗 Aventro Motors RAG Assistant")
st.caption("Ask a question about Aventro Motors documentation, or view the evaluation results.")


# ============================================================
# Session state
# ============================================================
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_docs" not in st.session_state:
    st.session_state.user_docs = []   # list of {"vector_store": ..., "count": int, "files": [...]}


# ============================================================
# Sidebar — page selector + custom keys + upload
# ============================================================
st.sidebar.header("🧭 Page")
page = st.sidebar.radio(
    "Select page",
    ["💬 Chat", "📊 Evaluation"],
    label_visibility="collapsed",
)

st.sidebar.divider()


# ---------- Custom API keys (optional) ----------
with st.sidebar.expander("🔑 Custom API Keys (optional)"):
    st.caption("Leave blank to use the shared demo keys.")
    user_groq = st.text_input("Your Groq API key", type="password", key="user_groq")
    user_google = st.text_input("Your Google API key", type="password", key="user_google")

    if st.button("🔄 Reload with these keys"):
        if user_groq:
            os.environ["GROQ_API_KEY"] = user_groq
        if user_google:
            os.environ["GOOGLE_API_KEY"] = user_google
        st.cache_resource.clear()
        st.rerun()


# ============================================================
# Chat page
# ============================================================
if page == "💬 Chat":
    st.sidebar.divider()
    st.sidebar.header("⚙️ Configuration")

    mode = st.sidebar.selectbox(
        "Pipeline",
        ["simple", "flexible", "advanced"],
        help="simple: strict context-only | flexible: allows general knowledge | advanced: adds citations",
    )
    top_k = st.sidebar.slider("Top K (chunks to retrieve)", 1, 10, 5)
    min_score = st.sidebar.slider("Min similarity score", 0.0, 1.0, 0.7, 0.05)

    if st.sidebar.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    # ---------- Document upload ----------
    st.sidebar.divider()
    st.sidebar.header("📎 Upload Documents")

    uploaded = st.sidebar.file_uploader(
        "Upload files to query alongside the main corpus",
        type=["pdf", "docx", "pptx", "txt", "md", "html", "htm", "csv"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    if uploaded and st.sidebar.button("📥 Index uploaded files"):
        with st.sidebar.status("Parsing and indexing...", expanded=False) as status:
            vs, n = build_user_collection(uploaded, st.session_state.session_id)
            if vs is not None:
                st.session_state.user_docs.append({
                    "vector_store": vs,
                    "count": n,
                    "files": [f.name for f in uploaded],
                })
                st.cache_resource.clear()
                status.update(label=f"✅ Indexed {n} chunks from {len(uploaded)} file(s).", state="complete")
            else:
                status.update(label="⚠️ No valid documents to index.", state="error")

    if st.session_state.user_docs:
        total = sum(d["count"] for d in st.session_state.user_docs)
        st.sidebar.info(f"📄 {total} user chunks indexed in this session")
        with st.sidebar.expander("📋 Uploaded files"):
            for d in st.session_state.user_docs:
                for fname in d["files"]:
                    st.write(f"- {fname}")
        if st.sidebar.button("🗑️ Clear uploads"):
            st.session_state.user_docs = []
            st.rerun()

    # ---------- Build retriever ----------
    embedding_manager, retriever_main, llm = init_rag()

    retrievers = [retriever_main]
    for doc in st.session_state.user_docs:
        retrievers.append(RAGRetriever(doc["vector_store"], embedding_manager))

    retriever = MultiRetriever(retrievers) if len(retrievers) > 1 else retriever_main

    # ---------- Render history ----------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"📚 Sources ({len(msg['sources'])})", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**[{i}] {src['source']}**")
                        page_val = src.get("page")
                        page_str = f"  •  Page: {page_val}" if page_val is not None and page_val != "unknown" else ""
                        st.caption(f"Similarity: {src['score']:.4f}{page_str}")
                        if src.get("preview"):
                            st.text(src["preview"])
                        st.divider()

    # ---------- Chat input ----------
    query = st.chat_input("Ask a question about Aventro Motors...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Retrieving and generating..."):
                    if mode == "simple":
                        result = rag_simple(query, retriever, llm, top_k=top_k, min_score=min_score)
                    elif mode == "flexible":
                        result = rag_flexible(query, retriever, llm, top_k=top_k, min_score=min_score)
                    elif mode == "advanced":
                        adv = AdvancedRAGPipeline(retriever, llm)
                        result = adv.query(query, top_k=top_k, min_score=min_score, summarize=False)

                st.markdown(result["answer"])

                sources = result.get("sources", [])
                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**[{i}] {src['source']}**")
                            page_val = src.get("page")
                            page_str = f"  •  Page: {page_val}" if page_val is not None and page_val != "unknown" else ""
                            st.caption(f"Similarity: {src['score']:.4f}{page_str}")
                            if src.get("preview"):
                                st.text(src["preview"])
                            st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": sources,
                })

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate" in err_str.lower():
                    error_msg = "⏱️ Rate limit reached. Please wait a few seconds and try again."
                else:
                    error_msg = f"An error occurred: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                })


# ============================================================
# Evaluation page
# ============================================================
elif page == "📊 Evaluation":
    st.header("📊 Evaluation Results")
    st.caption("Automated evaluation on 40 questions × 3 pipelines (120 runs), judged by an LLM-as-judge.")

    summary = load_evaluation_summary()

    if summary is None:
        st.warning(
            "No evaluation results found. Run `python evaluation/evaluate_rag.py` first, "
            "then refresh this page."
        )
    else:
        st.subheader("Summary by pipeline")

        rows = []
        for pipeline, data in sorted(summary["by_pipeline"].items()):
            rows.append({
                "Pipeline": pipeline,
                "In-domain accuracy": f"{data['in_domain']['accuracy']:.1%}",
                "Combined accuracy": f"{data['in_domain']['combined_accuracy']:.1%}",
                "Out-of-domain abstain": f"{data['out_of_domain']['abstain_rate']:.1%}",
                "Hallucination rate": f"{data['overall']['hallucination_rate']:.1%}",
                "Avg top score (in)": f"{data['in_domain']['avg_top_score']:.3f}",
                "Avg top score (out)": f"{data['out_of_domain']['avg_top_score']:.3f}",
                "Avg latency (s)": f"{data['overall']['avg_latency_sec']:.2f}",
            })

        df_summary = pd.DataFrame(rows)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

        st.subheader("Detailed counts")
        for pipeline, data in sorted(summary["by_pipeline"].items()):
            with st.expander(f"Pipeline: {pipeline}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**In-domain (20 questions)**")
                    ind = data["in_domain"]
                    st.write(f"- correct: {ind['correct']}/{ind['total']}")
                    st.write(f"- incomplete: {ind['incomplete']}/{ind['total']}")
                    st.write(f"- abstain: {ind['abstain']}/{ind['total']}")
                    st.write(f"- hallucination: {ind['hallucination']}/{ind['total']}")
                with col2:
                    st.markdown("**Out-of-domain (20 questions)**")
                    outd = data["out_of_domain"]
                    st.write(f"- correct: {outd['correct']}/{outd['total']}")
                    st.write(f"- abstain (expected): {outd['abstain']}/{outd['total']}")
                    st.write(f"- hallucination: {outd['hallucination']}/{outd['total']}")

        st.subheader("Per-question results")
        df = load_evaluation_csv()
        if df is not None:
            pipeline_filter = st.selectbox(
                "Filter by pipeline",
                ["all"] + sorted(df["pipeline"].unique().tolist()),
            )
            type_filter = st.selectbox(
                "Filter by type",
                ["all"] + sorted(df["type"].unique().tolist()),
            )
            filtered = df.copy()
            if pipeline_filter != "all":
                filtered = filtered[filtered["pipeline"] == pipeline_filter]
            if type_filter != "all":
                filtered = filtered[filtered["type"] == type_filter]

            st.dataframe(filtered, use_container_width=True, hide_index=True)

            csv_bytes = filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download filtered CSV",
                data=csv_bytes,
                file_name="filtered_results.csv",
                mime="text/csv",
            )

        st.subheader("Full report")
        report = load_evaluation_report()
        if report:
            with st.expander("View full Markdown report", expanded=False):
                st.markdown(report)