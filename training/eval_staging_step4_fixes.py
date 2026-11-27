"""Staging step 4, petchs only: two training-free fixes to the stage CNN B(b), measured on the petchs OOF.

* B(b)+U:    A-D says "U" -> U, otherwise B(b)'s argmax.
* B(b)+T4:   B(b) may say 4 only when p(4) >= t, otherwise the argmax over the other classes. t is tuned nested
             by source fold (5-class balanced accuracy on the other two folds, ties -> the smallest t).
* B(b)+U+T4: the rule-U override first, then T4 on the rest; t tuned nested with the override applied.

Only cached outputs are used (step-3 B(b) OOF probabilities, A-D feature cache). No model is retrained.
The locked external tests (PIID, kaggle) are never read.

    PYTHONPATH=. python -m training.eval_staging_step4_fixes   # -> runs/staging/step4/petchs_fixes.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from training.eval_staging_cv import bal, merge4
from training.eval_staging_petchs import CLASSES
from training.eval_staging_step3 import (
    FEAT_D,
    STEP3,
    b_oof,
    boot_samples,
    hybrid,
    nested_rules,
    report,
    show,
)

OUT = Path("runs/staging/step4/petchs_fixes.json")
T_GRID = [round(float(x), 2) for x in np.arange(0.30, 0.901, 0.05)]
I4 = CLASSES.index("4")


# ---------------------------------------------------------------- the fixes
def t4(prob: np.ndarray, t: float) -> np.ndarray:
    """Argmax, but a Stage-4 call needs p(4) >= t; otherwise the best of the other classes."""
    arg = prob.argmax(1)
    other = np.where(np.arange(len(CLASSES)) == I4, -np.inf, prob).argmax(1)
    idx = np.where((arg == I4) & (prob[:, I4] < t), other, arg)
    return np.array([CLASSES[i] for i in idx], object)


def u_override(rule_pred: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = pred.copy()
    out[rule_pred == "U"] = "U"
    return out


def fixed(prob: np.ndarray, t: float, rule_pred: np.ndarray | None) -> np.ndarray:
    p = t4(prob, t)
    return p if rule_pred is None else u_override(rule_pred, p)


def tune_t(prob: np.ndarray, y: np.ndarray, rule_pred: np.ndarray | None) -> tuple[float, dict]:
    """5-class balanced accuracy over the grid; ascending grid + strict '>' keeps the smallest t on ties."""
    curve = {t: round(bal(y, fixed(prob, t, rule_pred)), 12) for t in T_GRID}
    best = T_GRID[0]
    for t in T_GRID:
        if curve[t] > curve[best]:
            best = t
    return best, curve


def nested_t(prob: np.ndarray, y: np.ndarray, fold: np.ndarray,
             rule_pred: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    pred, ts = np.full(len(y), "", object), {}
    for k in range(3):
        tr, te = fold != k, fold == k
        ts[k], _ = tune_t(prob[tr], y[tr], None if rule_pred is None else rule_pred[tr])
        pred[te] = fixed(prob[te], ts[k], None if rule_pred is None else rule_pred[te])
    return pred, ts


# ---------------------------------------------------------------- extra scoring (existing functions untouched)
def ci(v) -> list[float]:
    return [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])]


def recall(y: np.ndarray, p: np.ndarray, c: str) -> float:
    return float((p[y == c] == c).mean()) if (y == c).any() else float("nan")


def extra(y: np.ndarray, preds: dict, groups: np.ndarray) -> dict:
    """U recall and Stage 4 sensitivity with the same cluster-bootstrap resamples as report()."""
    idx = boot_samples(groups)
    out = {}
    for m, p in preds.items():
        out[m] = {}
        for key, c in (("u_recall", "U"), ("stage4_sensitivity", "4")):
            out[m][key] = round(recall(y, p, c), 3)
            out[m][key + "_n"] = f"{int((p[y == c] == c).sum())}/{int((y == c).sum())}"
            out[m][key + "_ci95"] = ci([recall(y[i], p[i], c) for i in idx])
    return out


def nou_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """True-U images already removed. U / DTI / belirsiz predictions count as wrong (they never equal a stage)."""
    return {"bal_acc_1234": bal(y, p), "bal_acc_3class_1_2_34": bal(merge4(y), merge4(p)),
            "stage4_sensitivity": recall(y, p, "4")}


def nou_report(y: np.ndarray, preds: dict, groups: np.ndarray, base: str) -> dict:
    keep = y != "U"
    y, groups = y[keep], groups[keep]
    preds = {m: p[keep] for m, p in preds.items()}
    idx = boot_samples(groups)
    out: dict = {"n_images": int(keep.sum()), "label_counts": {c: int((y == c).sum()) for c in CLASSES[:4]},
                 "non_stage_predictions": {}, "methods": {}, "differences_vs_" + base: {}}
    boots = {m: [nou_metrics(y[i], p[i]) for i in idx] for m, p in preds.items()}
    for m, p in preds.items():
        s = nou_metrics(y, p)
        out["methods"][m] = {k: round(v, 3) for k, v in s.items()}
        out["methods"][m].update({k + "_ci95": ci([b[k] for b in boots[m]]) for k in s})
        out["non_stage_predictions"][m] = {k: int((p == k).sum()) for k in ("U", "DTI", "belirsiz")}
    for m in preds:
        if m == base:
            continue
        s, s0 = nou_metrics(y, preds[m]), nou_metrics(y, preds[base])
        out["differences_vs_" + base][f"{m} − {base}"] = {
            k: {"diff": round(s[k] - s0[k], 3), "ci95": ci([b[k] - b0[k] for b, b0 in zip(boots[m], boots[base],
                                                                                          strict=True)])}
            for k in s}
    return out


# ---------------------------------------------------------------- petchs OOF
def main() -> None:
    fd = json.loads(FEAT_D.read_text())
    names = sorted(fd)
    y = np.array([fd[n]["label"] for n in names], object)
    fold = np.array([fd[n]["fold"] for n in names])
    a_d, _ = nested_rules([fd[n]["D"] for n in names], y, fold)
    b = b_oof("bb")
    prob = np.array([b[n] for n in names])

    preds = {"A-D": a_d, "B(b)": np.array([CLASSES[i] for i in prob.argmax(1)], object), "H-b": hybrid(a_d, prob)}
    preds["B(b)+U"] = u_override(a_d, preds["B(b)"])
    preds["B(b)+T4"], ts_t4 = nested_t(prob, y, fold)
    preds["B(b)+U+T4"], ts_ut4 = nested_t(prob, y, fold, a_d)

    # the three step-3 rows must be reproduced exactly from the caches
    old = json.loads((STEP3 / "petchs_oof.json").read_text())["per_image"]
    for m in ("A-D", "B(b)", "H-b"):
        assert all(old[n][m] == preds[m][i] for i, n in enumerate(names)), m

    groups = np.array(names)  # one representative image per source, as in step 3
    fixes = ("B(b)+U", "B(b)+T4", "B(b)+U+T4")
    diffs = {}
    for f in fixes:
        diffs[f"{f} − B(b) (4-sınıf)"] = (f, "B(b)", "bal_acc_4")
        diffs[f"{f} − B(b) (5-sınıf)"] = (f, "B(b)", "bal_acc_5")
    for f in fixes:
        diffs[f"{f} − H-b (5-sınıf)"] = (f, "H-b", "bal_acc_5")
    diffs["B(b)+U+T4 − A-D (4-sınıf)"] = ("B(b)+U+T4", "A-D", "bal_acc_4")
    rep = report(y, preds, groups, diffs)

    rep["extra_metrics"] = extra(y, preds, groups)
    t_all, curve_all = tune_t(prob, y, None)
    tu_all, curveu_all = tune_t(prob, y, a_d)
    rep["t4_threshold"] = {
        "grid": T_GRID, "objective": "5-class balanced accuracy, ties -> smallest t",
        "B(b)+T4": {"per_fold": ts_t4, "all_petchs": t_all, "curve_all_petchs": curve_all},
        "B(b)+U+T4": {"per_fold": ts_ut4, "all_petchs": tu_all, "curve_all_petchs": curveu_all},
        "argmax_4_with_p4_below_0.30": int(((prob.argmax(1) == I4) & (prob[:, I4] < 0.30)).sum())}
    rep["no_U_preview"] = nou_report(y, preds, groups, "B(b)")
    rep["per_image"] = {n: {"label": str(y[i]), "fold": int(fold[i]), "p4": round(float(prob[i, I4]), 4),
                            **{m: str(p[i]) for m, p in preds.items()}} for i, n in enumerate(names)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1))

    show(rep)
    for m, s in rep["extra_metrics"].items():
        print(f"  {m:12s} U recall {s['u_recall_n']} {s['u_recall']:.3f} {s['u_recall_ci95']}  "
              f"Evre 4 {s['stage4_sensitivity_n']} {s['stage4_sensitivity_ci95']}")
    print("t:", {k: (v["per_fold"], v["all_petchs"]) for k, v in rep["t4_threshold"].items() if isinstance(v, dict)})
    nu = rep["no_U_preview"]
    print(f"U'suz önizleme: n = {nu['n_images']} {nu['label_counts']}")
    for m, s in nu["methods"].items():
        print(f"  {m:12s} 1-4 {s['bal_acc_1234']:.3f} {s['bal_acc_1234_ci95']}  3-sınıf {s['bal_acc_3class_1_2_34']:.3f} "
              f"{s['bal_acc_3class_1_2_34_ci95']}  Evre 4 {s['stage4_sensitivity']:.3f}  {nu['non_stage_predictions'][m]}")
    for k, v in nu["differences_vs_B(b)"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
