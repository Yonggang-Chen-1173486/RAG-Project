# Deliverables

This document lists the artefacts produced during the project, organised by
category. All code, data, and documentation are stored in the repository.

---

## 1. Source Code — Modular RAG Package

Refactored from a single exploratory notebook into a clean, installable
Python package.

**Location:** `src/rag/`

| Module | Purpose |
|---|---|
| `config.py` | Centralised configuration (paths, model names, chunk size, thresholds) |
| `embedding/embedding_manager.py` | Google Gemini embedding generation (`gemini-embedding-001`, 768-dim) |
| `ingestion/loaders.py` | Multi-format document loading (PDF, DOCX, PPTX, HTML, MD, TXT, CSV) |
| `ingestion/splitter.py` | Recursive text chunking |
| `ingestion/indexer.py` | Orchestrates embedding + storage |
| `retrieval/vector_store.py` | ChromaDB persistent vector store wrapper |
| `retrieval/retriever.py` | Query → ranked chunks with similarity scores |
| `generation/llm.py` | Groq `ChatGroq` wrapper (single `invoke()` interface) |
| `pipeline/simple_rag.py` | Context-only pipeline (strict abstain when context is insufficient) |
| `pipeline/flexible_rag.py` | Context + general-knowledge fallback |
| `pipeline/advanced_rag.py` | Citations, confidence, source previews, history |
| `app.py` (root) | Streamlit web interface (chat + evaluation dashboard) |

---

## 2. Command-Line Interface

**Location:** `scripts/`

| Script | Purpose |
|---|---|
| `ingest.py` | (Re)build or append to the ChromaDB vector store |
| `query.py` | Ask a question with `--mode`, `--top-k`, `--min-score`, `--log-level` |

### `query.py` Parameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `question` | positional | — | The question to ask |
| `--mode` | choice | `simple` | Pipeline: `simple`, `flexible`, or `advanced` |
| `--top-k` | int | `DEFAULT_TOP_K` from `config.py` | Number of chunks to retrieve |
| `--min-score` | float | `DEFAULT_SCORE_THRESHOLD` from `config.py` | Minimum similarity score to keep a chunk |
| `--log-level` | choice | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Examples — All Modes

    python scripts/query.py "When was Aventro Motors founded?"
    python scripts/query.py "What is the capital of France?" --mode flexible
    python scripts/query.py "Compare ACC and LKAS systems" --mode advanced

### Examples — Parameter Combinations

    python scripts/query.py "How does Adaptive Cruise Control work?" --top-k 6
    python scripts/query.py "What is the price of the Aventro Storm SUV?" --min-score 0.75
    python scripts/query.py "Compare ACC and LKAS systems" --mode advanced --top-k 6
    python scripts/query.py "List all safety features of the Aventro Grand SUV" --mode advanced --top-k 6 --min-score 0.7

### Examples — Logging Control

    python scripts/query.py "When was Aventro Motors founded?" --log-level WARNING
    python scripts/query.py "When was Aventro Motors founded?" --log-level DEBUG

### Ingestion

    python scripts/ingest.py
    python scripts/ingest.py --append
    python scripts/ingest.py --log-level DEBUG

---

## 2b. Web Interface

**Location:** project root

| File | Purpose |
|---|---|
| `app.py` | Streamlit web UI with chat interface and evaluation dashboard |

**Run:**

    streamlit run app.py

Then open http://localhost:8501.

**Features:**

- Chat interface with conversation history
- Pipeline selector (`simple` / `flexible` / `advanced`)
- Retrieval controls (`top_k`, `min_score`)
- Source viewer with similarity scores
- Evaluation dashboard (summary, per-question data, CSV download)

**Screenshot:** `deliverables/screenshots/chat.png`

---

## 3. Evaluation Suite

**Location:** `evaluation/`

| File | Purpose |
|---|---|
| `test_questions.json` | 40 test questions (20 in-domain, 20 out-of-domain) |
| `judge.py` | LLM-as-judge with four labels: `correct`, `incomplete`, `abstain`, `hallucination` |
| `evaluate_rag.py` | Runs 3 pipelines × 40 questions; produces CSV / JSON / Markdown reports |
| `results/results.csv` | Per-question detail (120 rows) |
| `results/summary.json` | Aggregated metrics |
| `results/report.md` | Human-readable evaluation report |

**Metrics produced:**

- In-domain accuracy, partial rate, combined accuracy
- Out-of-domain abstain rate
- Hallucination rate
- Average top similarity score (in-domain and out-of-domain)
- Average latency

**How to run:**

    python evaluation/evaluate_rag.py
    python evaluation/evaluate_rag.py --limit 5

---

## 4. Data

**Location:** `data/`

| Path | Description |
|---|---|
| `data/Aventro Motors/` | Source documents in 5 formats (DOCX, HTML, MD, PPTX, PDF) |
| `data/vector_store/` | Persistent ChromaDB collection |

The vector store is derived data and can be regenerated with:

    python scripts/ingest.py

---

## 5. Documentation

**Location:** `Deliverables/` and `References/` directories

| File | Purpose |
|---|---|
| `README.md` | Overview, installation, usage, project structure, configuration, roadmap |
| `DELIVERABLES.md` | This file |
| `REFERENCES.md` | Dataset, tutorial, libraries, models, and external services |
| `.gitignore` | Excludes `.env`, virtual environments, `__pycache__`, and `*.egg-info` |
| `app.py` | Streamlit web interface (chat + evaluation) |

---

## 6. Configuration and Packaging

| File | Purpose |
|---|---|
| `pyproject.toml` | Package metadata; supports `pip install -e .` |
| `requirements.txt` | Dependency list (mirrors `pyproject.toml`) |
| `uv.lock` | Locked dependency tree |
| `.env` (not committed) | Stores `GROQ_API_KEY` and `GOOGLE_API_KEY` |
| `.env` handling | Loaded at CLI entry points only (not inside library modules) |

---

## 7. Evaluation Results Summary

From the final run (40 questions × 3 pipelines = 120 runs):

| Pipeline | In-domain accuracy | Combined accuracy | Out-of-domain abstain | Hallucination rate |
|---|---|---|---|---|
| simple | 90.0% | 95.0% | 100.0% | 0.0% |
| flexible | 90.0% | 95.0% | 100.0% | 0.0% |
| advanced | 90.0% | 95.0% | 100.0% | 0.0% |

**Key findings:**

- Zero hallucination across all pipelines.
- Perfect abstain rate on out-of-domain questions (20/20 per pipeline).
- Two known limitations remain: one incomplete answer (`q003` — omitted the ECU component) and one abstain (`q015` — Aventro Grand SUV seating capacity), `q003` caused by different text understanding and `q015` caused by chunking granularity and topk selection rather than model failure.

Reproducible via:

    python evaluation/evaluate_rag.py

---

## 8. Notes on Project Evolution

- The original prototype lived in a single Jupyter notebook (`notebook/pdf_loader_old.ipynb`).
- Introduced a command-line interface (`ingest.py`, `query.py`).
- Produced the user-facing `README.md`, `DELIVERABLES.md`, and `REFERENCES.md`.
- Standardised logging, unified the LLM interface, removed dead code, and silenced third-party log noise.
- Added the evaluation suite described in Section 3.
- A follow-up iteration migrated the embedding layer from local `SentenceTransformers` (`all-MiniLM-L6-v2`, 384-dim) to Google's `gemini-embedding-001` (768-dim), re-tuned retrieval thresholds (`min_score=0.7`), hardened the pipeline prompts, and re-ran the full evaluation.
- The final phase added a **Streamlit web interface** (`app.py`) with chat history and an evaluation dashboard.

The system can be installed, ingested, queried, and evaluated without opening a notebook.