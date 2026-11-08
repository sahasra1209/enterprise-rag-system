import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "evaluation" / "results"
cases = {c["id"]: c for c in json.load(open(ROOT / "evaluation" / "dataset_financebench.json", encoding="utf-8"))}


def dump(run_file, only_verified=False):
    lines = [f"##### {run_file} #####"]
    for run in json.load(open(RES / run_file, encoding="utf-8")):
        c, r = cases[run["id"]], run["result"]
        claims = r.get("claims") or []
        if c["category"] != "supported" or not claims:
            continue
        if only_verified and r.get("overall_status") != "supported":
            continue
        chunks = {ch["citation_id"]: ch for ch in (r.get("retrieved_chunks") or [])}
        lines += ["", "=" * 80, f"{c['id']}   [{r.get('overall_status')}]",
                  "QUESTION : " + c["question"],
                  "REFERENCE: " + " ".join(c["reference_answer"].split()), "MODEL SAYS:"]
        for cl in claims:
            lines.append(f"  ({cl.get('status')}) {cl.get('claim') or '[empty]'}")
        cited = sorted({cid for cl in claims for cid in (cl.get("citations") or [])}) or list(chunks)
        for cid in cited:
            ch = chunks.get(cid)
            if ch:
                lines.append(f"  PASSAGE {cid} (page {ch['page']}): {' '.join(ch['text'].split())[:700]}")
    return lines


out = dump("run_prompt_d.json") + [""] + dump("run_prompt_c.json", only_verified=True)
(RES / "label_sheet.txt").write_text("\n".join(out), encoding="utf-8")
print("Wrote", RES / "label_sheet.txt")