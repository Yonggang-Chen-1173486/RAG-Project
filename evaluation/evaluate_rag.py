"""
Evaluate the Aventro Motors RAG system on a set of test questions.

Runs every question through three pipelines (simple, flexible, advanced),
judges each answer with an LLM-as-judge, and writes:
  - evaluation/results/results.csv      (per-question detail)
  - evaluation/results/summary.json     (aggregated metrics)
  - evaluation/results/report.md        (human-readable report)

Usage:
    python evaluation/evaluate_rag.py
    python evaluation/evaluate_rag.py --limit 5
    python evaluation/evaluate_rag.py --pipelines simple advanced
"""
import argparse
import csv
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

# Project setup
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore", module="huggingface_hub")


def _configure_logging():
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
    logging.getLogger("rag").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the RAG system.")
    p.add_argument(
        "--questions", type=str,
        default=str(_PROJECT_ROOT / "evaluation" / "test_questions.json"),
        help="Path to test_questions.json",
    )
    p.add_argument(
        "--out-dir", type=str,
        default=str(_PROJECT_ROOT / "evaluation" / "results"),
        help="Directory for output files",
    )
    p.add_argument(
        "--pipelines", nargs="+",
        default=["simple", "flexible", "advanced"],
        choices=["simple", "flexible", "advanced"],
        help="Which pipelines to evaluate",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N questions (for quick tests)",
    )
    p.add_argument(
        "--top-k", type=int, default=None,
        help="Override top_k (default: from config)",
    )
    p.add_argument(
        "--min-score", type=float, default=None,
        help="Override min_score (default: from config)",
    )
    return p.parse_args()


def load_questions(path: str, limit: int = None) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    if limit:
        questions = questions[:limit]
    return questions


def run_pipeline(
    pipeline_name: str,
    question: str,
    rag_simple, rag_flexible, AdvancedRAGPipeline,
    retriever, llm, top_k: int, min_score: float,
) -> Dict[str, Any]:
    """Run a single question through a single pipeline."""
    t0 = time.time()
    try:
        if pipeline_name == "simple":
            result = rag_simple(question, retriever, llm, top_k=top_k, min_score=min_score)
        elif pipeline_name == "flexible":
            result = rag_flexible(question, retriever, llm, top_k=top_k, min_score=min_score)
        elif pipeline_name == "advanced":
            adv = AdvancedRAGPipeline(retriever, llm)
            result = adv.query(question, top_k=top_k, min_score=min_score, summarize=False)
        else:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")
    except Exception as e:
        return {
            "answer": f"<ERROR: {e}>",
            "sources": [],
            "num_sources": 0,
            "top_score": 0.0,
            "latency_sec": time.time() - t0,
            "error": str(e),
        }

    sources = result.get("sources", [])
    top_score = max((s["score"] for s in sources), default=0.0)
    return {
        "answer": result.get("answer", ""),
        "sources": sources,
        "num_sources": len(sources),
        "top_score": top_score,
        "latency_sec": time.time() - t0,
        "error": None,
    }


def compute_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics per pipeline, split by in/out-of-domain."""
    summary: Dict[str, Any] = {"by_pipeline": {}}

    for pipeline in sorted(set(r["pipeline"] for r in rows)):
        pipe_rows = [r for r in rows if r["pipeline"] == pipeline]
        in_rows = [r for r in pipe_rows if r["type"] == "in_domain"]
        out_rows = [r for r in pipe_rows if r["type"] == "out_of_domain"]

        def counts(rs):
            return {
                "total": len(rs),
                "correct": sum(1 for r in rs if r["judge_label"] == "correct"),
                "incomplete": sum(1 for r in rs if r["judge_label"] == "incomplete"),
                "abstain": sum(1 for r in rs if r["judge_label"] == "abstain"),
                "hallucination": sum(1 for r in rs if r["judge_label"] == "hallucination"),
                "unknown": sum(1 for r in rs if r["judge_label"] == "unknown"),
            }

        def avg_score(rs):
            """Average top_score across all rows. Rows with no retrieved sources
            contribute 0 — this is the true retrieval performance."""
            if not rs:
                return 0.0
            return sum(r["top_score"] for r in rs) / len(rs)

        in_c = counts(in_rows)
        out_c = counts(out_rows)
        all_c = counts(pipe_rows)

        summary["by_pipeline"][pipeline] = {
            "in_domain": {
                **in_c,
                "accuracy": (in_c["correct"] / in_c["total"]) if in_c["total"] else 0.0,
                "partial_rate": (in_c["incomplete"] / in_c["total"]) if in_c["total"] else 0.0,
                "combined_accuracy": (
                    (in_c["correct"] + in_c["incomplete"]) / in_c["total"]
                    if in_c["total"] else 0.0
                ),
                "avg_top_score": avg_score(in_rows),
            },
            "out_of_domain": {
                **out_c,
                "abstain_rate": (out_c["abstain"] / out_c["total"]) if out_c["total"] else 0.0,
                "avg_top_score": avg_score(out_rows),
            },
            "overall": {
                **all_c,
                "hallucination_rate": (
                    all_c["hallucination"] / all_c["total"] if all_c["total"] else 0.0
                ),
                "avg_top_score": avg_score(pipe_rows),
                "avg_latency_sec": (
                    sum(r["latency_sec"] for r in pipe_rows) / len(pipe_rows)
                    if pipe_rows else 0.0
                ),
            },
        }

    return summary


def write_report(summary: Dict[str, Any], rows: List[Dict[str, Any]], path: Path, pipelines: List[str]):
    """Write a human-readable markdown report."""
    lines: List[str] = []
    lines.append("# RAG Evaluation Report\n")
    lines.append(f"- Pipelines: {', '.join(pipelines)}")
    lines.append(f"- Total runs: {len(rows)}")
    in_count = len([r for r in rows if r["pipeline"] == pipelines[0] and r["type"] == "in_domain"])
    out_count = len([r for r in rows if r["pipeline"] == pipelines[0] and r["type"] == "out_of_domain"])
    lines.append(f"- In-domain questions per pipeline: {in_count}")
    lines.append(f"- Out-of-domain questions per pipeline: {out_count}")
    lines.append("")

    lines.append("## Summary by pipeline\n")
    lines.append("| Pipeline | In-domain accuracy | In-domain partial | Combined | Out-of-domain abstain | Hallucination rate | Unknown rate | Avg top score (in) | Avg top score (out) | Avg latency (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for p in sorted(summary["by_pipeline"].keys()):
        s = summary["by_pipeline"][p]
        total_overall = s["overall"]["total"]
        unknown_rate = (s["overall"]["unknown"] / total_overall) if total_overall else 0.0
        lines.append(
            f"| {p} "
            f"| {s['in_domain']['accuracy']:.1%} "
            f"| {s['in_domain']['partial_rate']:.1%} "
            f"| {s['in_domain']['combined_accuracy']:.1%} "
            f"| {s['out_of_domain']['abstain_rate']:.1%} "
            f"| {s['overall']['hallucination_rate']:.1%} "
            f"| {unknown_rate:.1%} "
            f"| {s['in_domain']['avg_top_score']:.3f} "
            f"| {s['out_of_domain']['avg_top_score']:.3f} "
            f"| {s['overall']['avg_latency_sec']:.2f} |"
        )
    lines.append("")
    lines.append("> **Note:** `Avg top score` is the mean of `top_score` across all questions in each category. For out-of-domain questions, most retrievals are filtered by `min_score`, so the average is close to 0 — this correctly reflects that the system rejects irrelevant context.")
    lines.append("")

    lines.append("## Detailed counts\n")
    for p in sorted(summary["by_pipeline"].keys()):
        s = summary["by_pipeline"][p]
        lines.append(f"### Pipeline: `{p}`\n")
        lines.append("**In-domain**")
        lines.append(f"- correct: {s['in_domain']['correct']}/{s['in_domain']['total']}")
        lines.append(f"- incomplete: {s['in_domain']['incomplete']}/{s['in_domain']['total']}")
        lines.append(f"- abstain: {s['in_domain']['abstain']}/{s['in_domain']['total']}")
        lines.append(f"- hallucination: {s['in_domain']['hallucination']}/{s['in_domain']['total']}")
        lines.append(f"- avg top score: {s['in_domain']['avg_top_score']:.3f}")
        lines.append("")
        lines.append("**Out-of-domain**")
        lines.append(f"- correct (unexpected): {s['out_of_domain']['correct']}/{s['out_of_domain']['total']}")
        lines.append(f"- incomplete: {s['out_of_domain']['incomplete']}/{s['out_of_domain']['total']}")
        lines.append(f"- abstain (expected): {s['out_of_domain']['abstain']}/{s['out_of_domain']['total']}")
        lines.append(f"- hallucination: {s['out_of_domain']['hallucination']}/{s['out_of_domain']['total']}")
        lines.append(f"- avg top score: {s['out_of_domain']['avg_top_score']:.3f}")
        lines.append("")

    lines.append("## Hallucination cases\n")
    hallu = [r for r in rows if r["judge_label"] == "hallucination"]
    if not hallu:
        lines.append("_None._\n")
    else:
        lines.append("| ID | Pipeline | Question | Answer (truncated) |")
        lines.append("|---|---|---|---|")
        for r in hallu[:30]:
            lines.append(
                f"| {r['question_id']} | {r['pipeline']} "
                f"| {r['question'][:60]} "
                f"| {r['answer'][:120]} |"
            )
        if len(hallu) > 30:
            lines.append(f"\n_...and {len(hallu) - 30} more._")
        lines.append("")

    lines.append("## Incomplete cases (in-domain)\n")
    incomplete = [r for r in rows if r["judge_label"] == "incomplete" and r["type"] == "in_domain"]
    if not incomplete:
        lines.append("_None._\n")
    else:
        lines.append("| ID | Pipeline | Question | Judge reason |")
        lines.append("|---|---|---|---|")
        for r in incomplete[:30]:
            lines.append(
                f"| {r['question_id']} | {r['pipeline']} "
                f"| {r['question'][:50]} "
                f"| {r['judge_reason'][:120]} |"
            )
        if len(incomplete) > 30:
            lines.append(f"\n_...and {len(incomplete) - 30} more._")
        lines.append("")

    lines.append("## Abstain cases (in-domain)\n")
    in_abstain = [r for r in rows if r["judge_label"] == "abstain" and r["type"] == "in_domain"]
    if not in_abstain:
        lines.append("_None._\n")
    else:
        lines.append("| ID | Pipeline | Question | Top score | Judge reason |")
        lines.append("|---|---|---|---|---|")
        for r in in_abstain:
            lines.append(
                f"| {r['question_id']} | {r['pipeline']} "
                f"| {r['question'][:50]} "
                f"| {r['top_score']} "
                f"| {r['judge_reason'][:80]} |"
            )
        lines.append("")

    lines.append("## Abstain cases (out-of-domain)\n")
    out_abstain = [r for r in rows if r["type"] == "out_of_domain" and r["judge_label"] == "abstain"]
    lines.append(f"Total: {len(out_abstain)} cases where the model correctly abstained on out-of-domain questions.\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    _configure_logging()

    from rag.config import DEFAULT_TOP_K, DEFAULT_SCORE_THRESHOLD
    from rag.embedding.embedding_manager import EmbeddingManager
    from rag.retrieval.vector_store import VectorStore
    from rag.retrieval.retriever import RAGRetriever
    from rag.generation.llm import GroqLLM
    from rag.pipeline import rag_simple, rag_flexible, AdvancedRAGPipeline

    from evaluation.judge import judge_answer, _get_judge_llm

    top_k = args.top_k if args.top_k is not None else DEFAULT_TOP_K
    min_score = args.min_score if args.min_score is not None else DEFAULT_SCORE_THRESHOLD

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing RAG components...")
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()
    rag_retriever = RAGRetriever(vector_store, embedding_manager)
    groq_llm = GroqLLM()

    print("Initializing judge LLM...")
    judge_llm = _get_judge_llm()

    questions = load_questions(args.questions, args.limit)
    print(f"Loaded {len(questions)} questions. Pipelines: {args.pipelines}")
    print(f"top_k={top_k}, min_score={min_score}\n")

    rows: List[Dict[str, Any]] = []

    for i, q in enumerate(questions, start=1):
        qid = q["id"]
        qtype = q["type"]
        question_text = q["question"]
        expected = q.get("expected_answer")

        print(f"[{i}/{len(questions)}] {qid} ({qtype}): {question_text}")

        for pipeline_name in args.pipelines:
            run = run_pipeline(
                pipeline_name, question_text,
                rag_simple, rag_flexible, AdvancedRAGPipeline,
                rag_retriever, groq_llm,
                top_k, min_score,
            )

            if run["error"]:
                judge_result = {"label": "unknown", "reason": f"pipeline error: {run['error']}"}
            else:
                judge_result = judge_answer(question_text, expected, run["answer"], llm=judge_llm)

            rows.append({
                "question_id": qid,
                "type": qtype,
                "pipeline": pipeline_name,
                "question": question_text,
                "expected_answer": expected if expected is not None else "",
                "answer": run["answer"].replace("\n", " ")[:500],
                "top_score": round(run["top_score"], 4),
                "num_sources": run["num_sources"],
                "latency_sec": round(run["latency_sec"], 3),
                "judge_label": judge_result["label"],
                "judge_reason": judge_result["reason"],
            })

    csv_path = out_dir / "results.csv"
    fieldnames = [
        "question_id", "type", "pipeline", "question", "expected_answer",
        "answer", "top_score", "num_sources", "latency_sec",
        "judge_label", "judge_reason",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")

    summary = compute_summary(rows)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    report_path = out_dir / "report.md"
    write_report(summary, rows, report_path, args.pipelines)
    print(f"Wrote {report_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()