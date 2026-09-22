import csv

rows = list(csv.DictReader(open("evaluation/results/results.csv", encoding="utf-8")))

print("=== In-domain abstain cases ===")
for r in rows:
    if r["type"] == "in_domain" and r["judge_label"] == "abstain":
        print(f"\n[{r['question_id']}] {r['pipeline']}")
        print(f"  Q: {r['question']}")
        print(f"  A: {r['answer'][:200]}")
        print(f"  top_score: {r['top_score']}")
        print(f"  num_sources: {r['num_sources']}")
        print(f"  judge_reason: {r['judge_reason']}")

print("\n=== In-domain incomplete cases ===")
for r in rows:
    if r["type"] == "in_domain" and r["judge_label"] == "incomplete":
        print(f"\n[{r['question_id']}] {r['pipeline']}")
        print(f"  Q: {r['question']}")
        print(f"  A: {r['answer'][:200]}")
        print(f"  top_score: {r['top_score']}")
        print(f"  judge_reason: {r['judge_reason']}")