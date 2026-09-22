# RAG Evaluation Report

- Pipelines: simple, flexible, advanced
- Total runs: 120
- In-domain questions per pipeline: 20
- Out-of-domain questions per pipeline: 20

## Summary by pipeline

| Pipeline | In-domain accuracy | In-domain partial | Combined | Out-of-domain abstain | Hallucination rate | Unknown rate | Avg top score (in) | Avg top score (out) | Avg latency (s) |
|---|---|---|---|---|---|---|---|---|---|
| advanced | 90.0% | 5.0% | 95.0% | 100.0% | 0.0% | 0.0% | 0.855 | 0.144 | 4.18 |
| flexible | 90.0% | 5.0% | 95.0% | 100.0% | 0.0% | 0.0% | 0.855 | 0.144 | 4.20 |
| simple | 90.0% | 5.0% | 95.0% | 100.0% | 0.0% | 0.0% | 0.855 | 0.144 | 3.96 |

> **Note:** `Avg top score` is the mean of `top_score` across all questions in each category. For out-of-domain questions, most retrievals are filtered by `min_score`, so the average is close to 0 — this correctly reflects that the system rejects irrelevant context.

## Detailed counts

### Pipeline: `advanced`

**In-domain**
- correct: 18/20
- incomplete: 1/20
- abstain: 1/20
- hallucination: 0/20
- avg top score: 0.855

**Out-of-domain**
- correct (unexpected): 0/20
- incomplete: 0/20
- abstain (expected): 20/20
- hallucination: 0/20
- avg top score: 0.144

### Pipeline: `flexible`

**In-domain**
- correct: 18/20
- incomplete: 1/20
- abstain: 1/20
- hallucination: 0/20
- avg top score: 0.855

**Out-of-domain**
- correct (unexpected): 0/20
- incomplete: 0/20
- abstain (expected): 20/20
- hallucination: 0/20
- avg top score: 0.144

### Pipeline: `simple`

**In-domain**
- correct: 18/20
- incomplete: 1/20
- abstain: 1/20
- hallucination: 0/20
- avg top score: 0.855

**Out-of-domain**
- correct (unexpected): 0/20
- incomplete: 0/20
- abstain (expected): 20/20
- hallucination: 0/20
- avg top score: 0.144

## Hallucination cases

_None._

## Incomplete cases (in-domain)

| ID | Pipeline | Question | Judge reason |
|---|---|---|---|
| q003 | simple | What sensors does Adaptive Cruise Control use to d | The answer omitted the ECU component. |
| q003 | flexible | What sensors does Adaptive Cruise Control use to d | The answer omitted the ECU component. |
| q003 | advanced | What sensors does Adaptive Cruise Control use to d | The answer omitted the ECU component mentioned in the expected answer. |

## Abstain cases (in-domain)

| ID | Pipeline | Question | Top score | Judge reason |
|---|---|---|---|---|
| q015 | simple | How many seats does the Aventro Grand SUV have? | 0.8261 | Model states it cannot find the answer. |
| q015 | flexible | How many seats does the Aventro Grand SUV have? | 0.8261 | Model states it cannot find the answer in the context. |
| q015 | advanced | How many seats does the Aventro Grand SUV have? | 0.8261 | The model explicitly states it cannot find the answer. |

## Abstain cases (out-of-domain)

Total: 60 cases where the model correctly abstained on out-of-domain questions.
