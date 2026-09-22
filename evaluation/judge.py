"""
LLM-as-judge for the Aventro Motors RAG evaluation.

Classifies a model answer into one of four categories:
  - correct        : fully correct, covers all key facts
  - incomplete     : partially correct, omits key facts (but does not invent)
  - abstain        : the model explicitly says it cannot answer
  - hallucination  : the model invents facts not in the expected answer

Uses a different Groq model than the main pipeline (cross-model judging).
"""
import json
import os
import re
from typing import Dict, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

load_dotenv()

JUDGE_MODEL_CANDIDATES = [
    "openai/gpt-oss-20b",     # 首选：小、快、非 reasoning
    "openai/gpt-oss-120b",    # 次选：大、强
    "qwen/qwen3.8-27b",       # 兜底：和主模型相同，但确定可用
]


_JUDGE_PROMPT_TEMPLATE = """You are an impartial evaluator for a Retrieval-Augmented Generation (RAG) system.

You will receive:
- A question
- The expected answer (may be null if the question is out-of-domain)
- The model's actual answer

Classify the model's answer into EXACTLY ONE of these categories:

1. "correct"       - The answer is fully correct and covers all key facts in the expected answer.
2. "incomplete"    - The answer is partially correct but omits some key facts from the expected answer.
                     It does NOT invent new facts.
3. "abstain"       - The answer explicitly states that it cannot find the answer in the
                     knowledge base. If the model then offers general knowledge but
                     clearly labels it as such, this is still "abstain".
4. "hallucination" - The answer invents specific facts and presents them as if they
                     came from the retrieved knowledge base, without any disclaimer.
                     Contradicting the expected answer, or adding facts not in the
                     expected answer, also counts as "hallucination".

Special rules:

- If expected_answer is null (out-of-domain question):
  * "abstain" — the model clearly states it cannot find the answer in the knowledge base.
    IMPORTANT: Even if the model then provides general knowledge as a supplement, 
    as long as it FIRST acknowledges that the information is not from the knowledge 
    base, this still counts as "abstain". The key criterion is whether the model 
    disclaims the source.
  * "hallucination" — the model provides specific facts as if they came from the 
    knowledge base, WITHOUT any disclaimer. This is fabrication.

- If expected_answer is provided (in-domain question):
  * "correct" — the answer covers all key facts and is consistent with the expected 
    answer. Minor wording differences are fine. IMPORTANT: If the answer is consistent 
    with the expected answer, label it "correct" — do NOT label it "hallucination" 
    just because the wording differs or the answer is verbose.
  * "incomplete" — the answer covers only part of the expected answer, omitting 
    some key facts, without inventing anything.
  * "abstain" — the model says it cannot answer.
  * "hallucination" — the answer includes facts that CONTRADICT the expected answer, 
    or adds specific new facts NOT present in the expected answer.

Return ONLY a JSON object with these fields:
{{
  "label": "correct" | "incomplete" | "abstain" | "hallucination",
  "reason": "<one short sentence>"
}}

Question:
{question}

Expected answer:
{expected_answer}

Model's answer:
{model_answer}

JSON:"""


def _get_judge_llm() -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is required for the judge.")

    last_error = None
    for model_name in JUDGE_MODEL_CANDIDATES:
        try:
            llm = ChatGroq(
                groq_api_key=api_key,
                model_name=model_name,
                temperature=0.0,
                max_tokens=1000,
            )
            llm.invoke([HumanMessage(content="ping")])
            print(f"Judge using model: {model_name}")
            return llm
        except Exception as e:
            print(f"Model {model_name} unavailable: {e}")
            last_error = e

    raise ValueError(f"No judge model available. Last error: {last_error}")


def _parse_judge_response(text: str) -> Dict[str, str]:
    """Extract a JSON object from the judge's response (robust version)."""
    if not text or not text.strip():
        return {"label": "unknown", "reason": "Empty response from judge"}

    # 1) First try: non-greedy match of a simple JSON object
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            label = data.get("label", "unknown")
            if label not in ("correct", "incomplete", "abstain", "hallucination"):
                label = "unknown"
            return {"label": label, "reason": data.get("reason", "")}
        except json.JSONDecodeError:
            pass

    # 2) Second try: greedy match (handles nested braces)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            label = data.get("label", "unknown")
            if label not in ("correct", "incomplete", "abstain", "hallucination"):
                label = "unknown"
            return {"label": label, "reason": data.get("reason", "")}
        except json.JSONDecodeError:
            pass

    # 3) Last resort: search for the label keyword
    lower = text.lower()
    for candidate in ("hallucination", "abstain", "incomplete", "correct"):
        if f'"label": "{candidate}"' in lower or f"'label': '{candidate}'" in lower:
            return {"label": candidate, "reason": "(recovered from raw text)"}

    return {"label": "unknown", "reason": f"Could not parse: {text[:200]}"}


def judge_answer(
    question: str,
    expected_answer: Optional[str],
    model_answer: str,
    llm: Optional[ChatGroq] = None,
) -> Dict[str, str]:
    """
    Classify a model answer using an LLM-as-judge.

    Returns:
        {"label": "correct" | "incomplete" | "abstain" | "hallucination" | "unknown",
         "reason": str}
    """
    if llm is None:
        llm = _get_judge_llm()

    expected_str = (
        expected_answer if expected_answer is not None
        else "null (out-of-domain question)"
    )

    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        expected_answer=expected_str,
        model_answer=model_answer,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return _parse_judge_response(response.content)
    except Exception as e:
        return {"label": "unknown", "reason": f"Judge error: {e}"}