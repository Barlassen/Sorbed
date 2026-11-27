"""Post-hoc (not pre-registered) sensitivity check of the step-3 kaggle test after the crop-aware leak search.

``training.step4_crop_dups`` (SIFT + RANSAC, catches crops/rescales that dHash misses) found kaggle test images whose
copies sit in the step-3 training data (petchs dev set or fr7kn extra sources). Eye-review verdicts are in
``runs/staging/step3_kaggle_leak_review.json``. This re-scores the saved step-3 per-image predictions without those
images. No model is re-run and nothing new is measured; the original ``kaggle.json`` stays the pre-registered result.

    python -m training.step3_leak_posthoc
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from training.eval_staging_step3 import STEP3, metrics, report

CROP = Path("runs/staging/step4_crop_candidates.json")
REVIEW = Path("runs/staging/step3_kaggle_leak_review.json")
OUT = STEP3 / "kaggle_posthoc_leak.json"
DIFFS = {"karar 1 (birincil): B(b) − B(a)": ("B(b)", "B(a)", "bal_acc_4"),
         "karar 3 (kaggle): H − A-D": ("H", "A-D", "bal_acc_5"),
         "ikincil: H-b − A-D": ("H-b", "A-D", "bal_acc_5"),
         "bilgi: A-D − A × old-first": ("A-D", "A × old-first", "bal_acc_4")}


def main() -> None:
    crop = json.loads(CROP.read_text())
    verdict = json.loads(REVIEW.read_text())
    pairs = crop["kaggle_leak"] + [{"id": f"nt{i}", "kaggle": r["query"], "other": r["other"],
                                    "set": r["set"].removeprefix("kk_")}
                                   for i, r in enumerate(crop["near_threshold"]) if r["set"].startswith("kk_")]
    leaked: dict[str, set[str]] = {}
    for c in pairs:
        if verdict.get(c["id"], {}).get("verdict") in ("same", "same_wound"):
            leaked.setdefault(c["kaggle"], set()).add(c["set"])
    rows = [r for r in json.loads((STEP3 / "kaggle.json").read_text())["per_image"] if r["label"] != "SDTI"]
    methods = [k for k in rows[0] if k not in ("file", "label", "source")]
    keep = np.array([r["file"] not in leaked for r in rows])
    y = np.array([r["label"] for r in rows], object)
    groups = np.array([r["source"] for r in rows], object)
    preds = {m: np.array([r[m] for r in rows], object) for m in methods}
    rep = report(y[keep], {m: p[keep] for m, p in preds.items()}, groups[keep], DIFFS)
    rep["posthoc"] = "not pre-registered; leaked images removed, saved step-3 predictions re-scored"
    rep["leaked_primary_images"] = {f: sorted(s) for f, s in sorted(leaked.items()) if f in {r["file"] for r in rows}}
    rep["leaked_subset"] = {m: {k: round(v, 3) for k, v in metrics(y[~keep], p[~keep]).items()} for m, p in preds.items()}
    rep["leaked_subset_n"] = int((~keep).sum())
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(f"kept {keep.sum()} / {len(rows)} primary images ({rep['n_sources']} sources); leaked {(~keep).sum()}")
    for m, v in rep["methods"].items():
        print(f"  {m:14s} 4-sınıf {v['bal_acc_4']:.3f} {v['bal_acc_4_ci95']}  5-sınıf {v['bal_acc_5']:.3f} "
              f"{v['bal_acc_5_ci95']}  E4 duy. {v['stage4_sensitivity']:.2f}   | sızan alt küme 4-sınıf "
              f"{rep['leaked_subset'][m]['bal_acc_4']:.3f}")
    for k, v in rep["differences"].items():
        print(f"  {k}: {v['diff']:+.3f} {v['ci95']}")


if __name__ == "__main__":
    main()
