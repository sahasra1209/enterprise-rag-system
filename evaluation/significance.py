import json
from pathlib import Path
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
rows = json.load(open(ROOT / "evaluation" / "results" / "retrieval_experiment_scoped.json"))


def rr(rank):
    return 1 / rank if rank else 0.0


def hit3(rank):
    return bool(rank and rank <= 3)


v = [rr(r["vector"]) for r in rows]
k = [rr(r["rerank20"]) for r in rows]
better = sum(b > a for a, b in zip(v, k))
worse = sum(b < a for a, b in zip(v, k))
print(f"Reciprocal rank, rerank20 vs vector: better={better} worse={worse} tied={len(rows) - better - worse}")
if better + worse:
    print("Wilcoxon p =", round(wilcoxon(v, k, zero_method="wilcox").pvalue, 3))

gain = sum(1 for r in rows if not hit3(r["vector"]) and hit3(r["rerank20"]))
loss = sum(1 for r in rows if hit3(r["vector"]) and not hit3(r["rerank20"]))
print(f"Hit@3: gained={gain} lost={loss}")
if gain + loss:
    print("Exact sign test p =", round(binomtest(gain, gain + loss, 0.5).pvalue, 3))