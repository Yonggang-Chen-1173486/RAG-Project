# References

This document lists the external resources used during the development of this
project. It covers the source dataset, the tutorial that inspired the original
prototype, and the libraries, models, and tools that the system depends on.

---

## 1. Dataset

**Aventro Motors multi-format corpus**

- **Source:** https://github.com/udayallu/RAG-Multi-Corpus/tree/main/datasets
- **Formats used:** PDF, DOCX, PPTX, HTML, Markdown, CSV
- **Description:** A fictional Indian automobile manufacturer ("Aventro Motors")
  with documentation covering company profile, vehicle models, pricing, safety
  features, service procedures, FAQs, and more.
- **Purpose in this project:** Provides the ingestion corpus for the RAG system
  and the factual basis for the in-domain evaluation questions.

> Note: The dataset is used strictly for academic coursework.

---

## 2. Foundational Learning Material

**Original YouTube tutorial that inspired the prototype**

- **Source:** https://www.youtube.com/watch?v=o126p1QN_RI
- **Series:** RAG tutorial playlist
- **Description:** An introductory walkthrough of building a RAG pipeline in
  Python using LangChain, a vector store, and an LLM.
- **How it was used:** Served as the starting point for the original notebook
  prototype. The final system has since been extended considerably — see
  Section 7 for the differences.

---

## 3. Core Frameworks and Libraries

| Library | Role |
|---|---|
| `langchain-community` | Document loaders for PDF, DOCX, PPTX, HTML, MD, TXT, CSV |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` for chunking |
| `langchain-core` | `Document`, `HumanMessage`, `PromptTemplate` |
| `langchain-groq` | `ChatGroq` LLM interface |
| `langchain-google-genai` | Google Gemini embeddings integration |
| `chromadb` | Persistent vector store with cosine similarity search |
| `python-dotenv` | Loading environment variables from `.env` |
| `numpy` | Vector maths |
| `pytest` | Unit testing framework (development dependency) |
| `streamlit` | Unit testing framework (development dependency) |
| `pytest` | Web UI |

---

## 4. Models

**Embedding model**

- **Name:** `gemini-embedding-001`
- **Provider:** Google Generative AI (Gemini)
- **Dimensions:** 768
- **Runs:** Cloud API (requires `GOOGLE_API_KEY`)
- **Purpose:** Converts document chunks and user queries into dense vectors.

**Primary LLM**

- **Name:** `Qwen/Qwen3.8-27B`
- **Provider:** Groq (cloud API)
- **Purpose:** Generates answers from retrieved context in all three pipelines.

**Judge LLM (used for evaluation only)**

- **Candidates:** `openai/gpt-oss-20b`, `openai/gpt-oss-120b`
- **Provider:** Groq (cloud API)
- **Purpose:** An LLM-as-judge that classifies each answer into one of four
  categories: `correct`, `incomplete`, `abstain`, or `hallucination`.
- **Design note:** A different model from the primary LLM is preferred, to
  reduce self-evaluation bias.

---

## 5. External Services

| Service | Purpose |
|---|---|
| **Groq** (https://console.groq.com) | Hosted LLM inference (primary and judge) |
| **Google AI Studio** (https://aistudio.google.com) | API key for the Gemini embedding model |

---

## 6. Development Tools

| Tool | Role |
|---|---|
| Python 3.13 | Runtime |
| `pip` + `pyproject.toml` | Packaging (`pip install -e .`) |
| `uv` | Dependency resolution and lockfile management |
| Jupyter / VS Code | Development environment |
| Git + GitHub | Version control |
| `logging` (standard library) | Structured logging across all modules |
| `Streamlit` | Web UI framework (app.py) |
| `pandas` | Data display in evaluation dashboard |

---

## 7. How This Project Extends the Original Tutorial

The original YouTube tutorial provided a minimal RAG skeleton. The following
components were designed and implemented independently as part of this project:

- **Multi-format ingestion** — the tutorial focused on a single format;
  this project supports PDF, DOCX, PPTX, HTML, MD, TXT, and CSV.
- **Modular package structure** — the tutorial used a single script; this
  project is organised into an installable Python package (`src/rag/`).
- **Three distinct pipelines** — `simple`, `flexible`, and `advanced` — each
  with different behaviour when relevant context is not found.
- **Centralised configuration** — all tunable parameters live in `config.py`.
- **Command-line interface** — `scripts/ingest.py` and `scripts/query.py`
  with `--mode`, `--top-k`, `--min-score`, and `--log-level` options.
- **Structured logging** — a `--log-level` flag replaces scattered `print`
  calls and silences third-party library noise.
- **Cloud embedding migration** — moved from a local 384-dim model
  (`all-MiniLM-L6-v2`) to Google's 768-dim `gemini-embedding-001` for stronger
  semantic retrieval.
- **Evaluation suite** — a 40-question test set (20 in-domain, 20 out-of-domain)
  and an LLM-as-judge that measures accuracy, abstain rate, and hallucination
  rate across all three pipelines.
- **Unified LLM interface** — all pipelines call a single `GroqLLM.invoke()`
  method instead of interacting with `ChatGroq` directly.
- **Prompt hardening** — `simple` and `advanced` pipelines were rewritten to
  forbid using general knowledge, matching the strict behaviour of the
  `flexible` pipeline.
- **Interactive web interface** — a Streamlit app (`app.py`) with a chat UI,
  conversation history, pipeline selector, source viewer, and an evaluation
  dashboard, enabling non-CLI usage of the RAG system.

---

## 8. Citation Format

If referencing this project academically:

> *Aventro Motors RAG System.* Coursework project, COMP693.
> Dataset adapted from `udayallu/RAG-Multi-Corpus`.
> Original prototype inspired by an online RAG tutorial series.