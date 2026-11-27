"""Staging step 4: the frozen step-3 methods on PIID, the second locked external test (run ONCE).

Pre-registered in HANDOFF ("ÖN-KAYIT — evreleme adım 4", approved 2026-11-27). Nothing is retrained or re-tuned: rule
thresholds are the step-3 all-petchs values, B = mean of the 3 folds x 3 seeds, D mask = mean of the 3 fold DeepLabs
> 0.5, tissue = mean of the 3 fold tissue CNNs; the inference loop is ``eval_staging_step3.kaggle()`` with the PIID
list (``runs/staging/step4_lists.json``) in place of kaggle.

PIID has stages 1-4 only (no U, no SDTI). Primary metric: 4-stage balanced accuracy over {1, 2, 3, 4}; a U, DTI or
"belirsiz" output counts as wrong. Secondary: 3-class {1, 2, 3/4}, Stage 4 sensitivity, share of non-stage outputs.
Decisions: K1 B(b) - B(a), K2 H - A-D (secondary H-b - A-D), K3 B(b) - A-D (information only), all 4-stage.

    python -m training.eval_staging_step4_piid

Raw per-image outputs are cached in ``runs/staging/step4/piid_raw.json`` (resumable); the scored report goes to
``runs/staging/step4/piid.json`` and the script refuses to score again once that file exists.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from training.eval_staging_cv import bal, columns, merge4, rules_fast
from training.eval_staging_petchs import CLASSES, features
from training.eval_staging_step3 import (
    FEAT_D,
    FEAT_OLD,
    STEP3,
    boot_samples,
    hybrid,
    tissue_avg,
    tune3,
)

STEP4 = Path("runs/staging/step4")
LISTS = Path("runs/staging/step4_lists.json")
RAW = STEP4 / "piid_raw.json"
OUT = STEP4 / "piid.json"
STAGES = ("1", "2", "3", "4")
COLS = (*CLASSES, "DTI", "belirsiz")
DIFFS = {"K1 (birincil): B(b) − B(a)": ("B(b)", "B(a)"),
         "K2: H − A-D": ("H", "A-D"),
         "K2 ikincil: H-b − A-D": ("H-b", "A-D"),
         "K3 (bilgi): B(b) − A-D": ("B(b)", "A-D")}


def frozen_thresholds() -> tuple[dict, dict]:
    """Re-derive the step-3 all-petchs thresholds and check them against the values stored with the kaggle run."""
    fd, fo = json.loads(FEAT_D.read_text()), json.loads(FEAT_OLD.read_text())
    names = sorted(fd)
    y_p = np.array([fd[n]["label"] for n in names], object)
    th_d = tune3(columns([fd[n]["D"] for n in names]), y_p)
    th_old = tune3(columns([fo[n]["R2"] for n in names]), y_p)
    stored = json.loads((STEP3 / "kaggle.json").read_text())["thresholds_all_petchs"]
    assert stored == {"A × old-first": th_old, "A-D": th_d}, (stored, th_old, th_d)
    return th_d, th_old


def infer(rows: list[dict]) -> dict:
    from ultralytics import YOLO

    from sorbed.preprocess.color_norm import gray_world_normalize
    from training.eval_layered_map import deeplab_prob
    from training.tissue_ab_compare import SEG_NEW, SEG_OLD, wound_mask
    from training.train_stage_cnn import StageNet, device, predict
    from training.train_tissue_cnn import load as load_cnn
    from training.train_wound_deeplab import build_model
    from training.train_wound_deeplab import load as load_deeplab

    raw = json.loads(RAW.read_text()) if RAW.exists() else {}
    todo = [r for r in rows if r["file"] not in raw]
    if not todo:
        return raw
    dls = [load_deeplab(f"runs/wound_deeplab/fold{k}.pt") for k in range(3)]
    cnns = [load_cnn(f"runs/tissue_cnn_fold/fold{k}.pt") for k in range(3)]
    assert len({m.n_cls for m in cnns}) == 1
    segs = (YOLO(SEG_OLD), YOLO(SEG_NEW))
    b_models = {}
    for arm in ("ba", "bb"):
        b_models[arm] = []
        for k, s in itertools.product(range(3), range(3)):
            m = StageNet(build_model(pretrained=False).backbone)
            m.load_state_dict(torch.load(STEP3 / f"{arm}_fold{k}_s{s}.pt", map_location="cpu"))
            b_models[arm].append(m.to(device()).eval())

    STEP4.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(todo):
        img = cv2.imread(r["file"])
        rgb = gray_world_normalize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        d = np.mean([deeplab_prob(m, img) for m in dls], 0) > 0.5
        old = wound_mask(segs, img, "old-first")
        raw[r["file"]] = {
            "D": dataclasses.asdict(features(rgb, d, tissue_avg(cnns, img, d))),
            "old": dataclasses.asdict(features(rgb, old, tissue_avg(cnns, img, old))),
            **{arm: np.mean([predict(m, [img])[0] for m in ms], 0).tolist() for arm, ms in b_models.items()}}
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            RAW.write_text(json.dumps(raw))
            print(f"{len(raw)} / {len(rows)}", flush=True)
    return raw


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"bal_acc_stage4": bal(y, p), "bal_acc_3": bal(merge4(y), merge4(p)),
            "stage4_sensitivity": float((p[y == "4"] == "4").mean()),
            "nonstage_share": float((~np.isin(p, STAGES)).mean())}


def score(rows: list[dict], raw: dict, th_d: dict, th_old: dict) -> dict:
    y = np.array([r["label"] for r in rows], object)
    groups = np.array([r["source"] for r in rows], object)
    probs = {arm: np.array([raw[r["file"]][arm] for r in rows]) for arm in ("ba", "bb")}
    preds = {"hep Evre 3": np.full(len(y), "3", object),
             "A × old-first": rules_fast(columns([raw[r["file"]]["old"] for r in rows]), th_old),
             "A-D": rules_fast(columns([raw[r["file"]]["D"] for r in rows]), th_d)}
    for arm, tag in (("ba", "B(a)"), ("bb", "B(b)")):
        preds[tag] = np.array([CLASSES[i] for i in probs[arm].argmax(1)], object)
    preds["H"] = hybrid(preds["A-D"], probs["ba"])
    preds["H-b"] = hybrid(preds["A-D"], probs["bb"])

    idx = boot_samples(groups)
    ci = lambda v: [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])]  # noqa: E731
    rep: dict = {"n_images": len(y), "n_sources": len(np.unique(groups)),
                 "label_counts": {c: int((y == c).sum()) for c in STAGES}, "methods": {}, "differences": {}}
    for m, p in preds.items():
        s = metrics(y, p)
        boots = [metrics(y[i], p[i]) for i in idx]
        rep["methods"][m] = {k: round(v, 3) for k, v in s.items()}
        for k in ("bal_acc_stage4", "bal_acc_3", "stage4_sensitivity"):
            rep["methods"][m][k + "_ci95"] = ci([b[k] for b in boots])
        rep["methods"][m]["confusion"] = {c: {k: int(((y == c) & (p == k)).sum()) for k in COLS} for c in STAGES}
        staged = np.isin(p, STAGES)  # K3 descriptive: accuracy where the method gave a stage
        rep["methods"][m]["staged_only"] = {"n": int(staged.sum()),
                                            "bal_acc_stage4": round(bal(y[staged], p[staged]), 3) if staged.any() else None}
    for name, (a, b) in DIFFS.items():
        d = [bal(y[i], preds[a][i]) - bal(y[i], preds[b][i]) for i in idx]
        diff = bal(y, preds[a]) - bal(y, preds[b])
        lo, hi = ci(d)
        rep["differences"][name] = {"metric": "bal_acc_stage4", "diff": round(diff, 3), "ci95": [lo, hi],
                                    "met": bool(lo > 0) if not name.startswith("K3") else None}
    rep["thresholds_all_petchs"] = {"A × old-first": th_old, "A-D": th_d}
    rep["per_image"] = [{"file": r["file"], "label": r["label"], "source": r["source"],
                         **{m: str(p[i]) for m, p in preds.items()}} for i, r in enumerate(rows)]
    return rep


def show(rep: dict) -> None:
    print(f"n = {rep['n_images']} görüntü / {rep['n_sources']} kaynak, {rep['label_counts']}")
    for m, v in rep["methods"].items():
        print(f"  {m:14s} 4-evre {v['bal_acc_stage4']:.3f} {v['bal_acc_stage4_ci95']}  3-sınıf {v['bal_acc_3']:.3f} "
              f"{v['bal_acc_3_ci95']}  E4 duy. {v['stage4_sensitivity']:.2f}  evre dışı {v['nonstage_share']:.2f}  "
              f"evre verdiğinde {v['staged_only']}")
    for k, v in rep["differences"].items():
        print(f"  {k}: {v['diff']:+.3f} {v['ci95']} met={v['met']}")


def main() -> None:
    if OUT.exists():
        raise SystemExit("PIID was already measured once; the pre-registration allows a single run")
    th_d, th_old = frozen_thresholds()
    rows = json.loads(LISTS.read_text())["piid"]
    assert len(rows) == 1052 and len({r["source"] for r in rows}) == 1029
    raw = infer(rows)
    rep = score(rows, raw, th_d, th_old)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    show(rep)


if __name__ == "__main__":
    main()
