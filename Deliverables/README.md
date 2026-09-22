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
| `embedding/embedding_manager.py` | Embedding generation via Google Gemini (`gemini-embedding-001`) |
| `ingestion/loaders.py` | Multi-format document loading (PDF, DOCX, PPTX, HTML, MD, TXT, CSV) |
| `ingestion/splitter.py` | Recursive text chunking |
| `ingestion/indexer.py` | Orchestrates embedding + storage |
| `retrieval/vector_store.py` | ChromaDB persistent vector store wrapper |
| `retrieval/retriever.py` | Query → ranked chunks with similarity scores |
| `generation/llm.py` | Groq `ChatGroq` wrapper (single `invoke()` interface) |
| `pipeline/simple_rag.py` | Context-only pipeline |
| `pipeline/flexible_rag.py` | Context + general-knowledge fallback |
| `pipeline/advanced_rag.py` | Citations, confidence, history |

---

## 2. Command-Line Interface

**Location:** `scripts/`

| Script | Purpose |
|---|---|
| `ingest.py` | (Re)build or append to the ChromaDB vector store |
| `query.py` | Ask a question with `--mode`, `--top-k`, `--min-score`, `--log-level` |

**Simple mode (default): context-only answer**

    python scripts/query.py "When was Aventro Motors founded?"

**Flexible mode: falls back to general knowledge when no context is found**

    python scripts/query.py "What is the capital of France?" --mode flexible

**Advanced mode: answer with citations, sources, and confidence scores**

    python scripts/query.py "Compare ACC and LKAS systems" --mode advanced

### Examples — Parameter Combinations

**Advanced mode with more context:**

    python scripts/query.py "Compare ACC and LKAS systems" --mode advanced --top-k 7

**Flexible mode with a stricter threshold:**

    python scripts/query.py "What is quantum computing?" --mode flexible --min-score 0.3

**All parameters combined:**

    python scripts/query.py "List all safety features of the Aventro Grand SUV" --mode advanced --top-k 7 --min-score 0.25

---

## 3. Evaluation Suite

**Location:** `evaluation/`

| File | Purpose |
|---|---|
| `test_questions.json` | 40 test questions (20 in-domain, 20 out-of-domain) |
| `judge.py` | LLM-as-judge with four labels: `correct`, `incomplete`, `abstain`, `hallucination` |
| `evaluate_rag.py` | Runs 3 pipelines × 40 questions, produces CSV / JSON / Markdown reports |
| `results/results.csv` | Per-question detail (120 rows) |
| `results/summary.json` | Aggregated metrics |
| `results/report.md` | Human-readable evaluation report |

**Key metrics produced:**

- In-domain accuracy, partial rate, combined accuracy
- Out-of-domain abstain rate
- Hallucination rate
- Average top similarity score and latency per pipeline

**How to run:**

    python evaluation/evaluate_rag.py
    python evaluation/evaluate_rag.py --limit 5

---

## 4. Data

**Location:** `data/`

| Path | Description |
|---|---|
| `data/Aventro Motors/` | Source documents in 5 formats (DOCX, HTML, MD, PPTX, PDF) |
| `data/vector_store/` | Persistent ChromaDB collection (42 chunks) |

The vector store is derived data and can be regenerated with:

    python scripts/ingest.py

---

## 5. Documentation

**Location:** project root

| File | Purpose |
|---|---|
| `README.md` | Overview, installation, usage, project structure, configuration, roadmap |
| `DELIVERABLES.md` | This file |
| `.gitignore` | Excludes `.env`, virtual environments, `__pycache__`, and `*.egg-info` |

---

## 6. Configuration and Packaging

| File | Purpose |
|---|---|
| `pyproject.toml` | Package metadata; supports `pip install -e .` |
| `.env` (not committed) | Stores `GROQ_API_KEY` |
| `.env` handling | Loaded at CLI entry points only (not inside library modules) |

---

## 7. Evaluation Results Summary

From the final run (40 questions × 3 pipelines = 120 runs):

| Pipeline | In-domain accuracy | Combined accuracy | Out-of-domain abstain | Hallucination rate |
|---|---|---|---|---|
| simple | 80.0% | 85.0% | 100.0% | 2.5% |
| flexible | 80.0% | 85.0% | 100.0% | 2.5% |
| advanced | 85.0% | 90.0% | 100.0% | 2.5% |

These results are reproducible via:

    python evaluation/evaluate_rag.py

---

## 8. Notes on Project Evolution

- The original prototype lived in a single Jupyter notebook (`notebook/pdf_loader_old.ipynb`).
- Direction A introduced a command-line interface.
- Direction D produced the user-facing `README.md`.
- Direction C standardised logging, unified LLM calls, and removed dead code.
- Direction B added the evaluation suite described in Section 3.

All four directions are complete. The system can be installed, ingested,
queried, and evaluated without opening a notebook.