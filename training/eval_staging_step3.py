"""Staging step 3: more data for the stage CNN, rules on the DeepLab D mask, and a rules+CNN hybrid.

Pre-registered in HANDOFF (2026-11-04). Arms:

* A x old-first: step-2 rule engine on the old-first mask (thresholds re-tuned with erythema fixed at 0.15).
* A-D: the same engine and grid on the source-fold DeepLab D mask with that fold's tissue CNN.
* B(a) / B(b): stage CNN on petchs only / petchs + fr7kn extra sources (``run_step3_b.sh``), 3 seeds averaged.
* H / H-b: A-D, but "Stage 3" becomes "Stage 4" when B(a) / B(b) gives p(4) > p(3).

    python -m training.eval_staging_step3 --features   # A-D features on petchs (cached)
    python -m training.eval_staging_step3 --petchs     # petchs OOF, n = 166 sources
    python -m training.eval_staging_step3 --kaggle     # locked external test: run ONCE, after everything is frozen
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from training.eval_staging_cv import GRID, bal, columns, engine_stage, merge4, rules_fast
from training.eval_staging_petchs import CLASSES, features

STEP3 = Path("runs/staging/step3")
FEAT_D = STEP3 / "features_D.json"
FEAT_OLD = Path("runs/staging/cv_features.json")  # step 2 cache: R2 = old-first x fold tissue CNN
LISTS = Path("runs/staging/step3_lists.json")
GRID3 = {**GRID, "e": [0.15]}  # erythema fixed (step-2 safety deviation)
DEFAULT3 = {"u": 0.5, "g": 0.15, "d": 0.1, "open": 0.15, "e": 0.15}
N_BOOT = 10_000


# ---------------------------------------------------------------- features
def extract_d() -> dict:
    from sorbed.preprocess.color_norm import gray_world_normalize
    from training.eval_layered_map import layers
    from training.eval_staging_petchs import items
    from training.petchs_groups import SPLITS, eval_representatives, source, source_folds
    from training.train_tissue_cnn import load as load_cnn
    from training.train_wound_deeplab import load as load_deeplab

    folds, reps = source_folds(), eval_representatives()
    dls = [load_deeplab(f"runs/wound_deeplab/fold{k}.pt") for k in range(3)]
    cnns = [load_cnn(f"runs/tissue_cnn_fold/fold{k}.pt") for k in range(3)]
    out = {}
    for split in SPLITS:
        for name, img, _, label in items(split):
            if name not in reps:
                continue
            k = folds[source(name)]
            lay = layers(img, dls[k], cnns[k])
            rgb = gray_world_normalize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            lbl = lay["lbl"] if lay["D"].sum() >= 30 else None  # as step 2: colour model on tiny regions
            out[name] = {"label": label, "fold": k, "D": dataclasses.asdict(features(rgb, lay["D"], lbl))}
            print(len(out), name, label, flush=True)
    STEP3.mkdir(parents=True, exist_ok=True)
    FEAT_D.write_text(json.dumps(out))
    return out


# ---------------------------------------------------------------- rules, hybrid, scoring
def tune3(X: dict, y: np.ndarray) -> dict:
    """Step-2 grid search (4-class balanced accuracy, ties -> closest to the defaults), erythema fixed."""
    best, best_key = None, None
    for vals in itertools.product(*GRID3.values()):
        th = dict(zip(GRID3, vals, strict=True))
        key = (bal(merge4(y), merge4(rules_fast(X, th))),
               -sum(abs(th[k] - DEFAULT3[k]) / DEFAULT3[k] for k in DEFAULT3))
        if best_key is None or key > best_key:
            best, best_key = th, key
    return best


def nested_rules(rows: list[dict], y: np.ndarray, fold: np.ndarray) -> tuple[np.ndarray, dict]:
    """Thresholds tuned on the other two folds, applied to the held-out one; checked against the real engine."""
    X = columns(rows)
    pred, ths = np.full(len(y), "", object), {}
    for k in range(3):
        tr, te = fold != k, fold == k
        ths[k] = tune3({c: v[tr] for c, v in X.items()}, y[tr])
        pred[te] = rules_fast({c: v[te] for c, v in X.items()}, ths[k])
        for i in np.where(te)[0]:
            assert engine_stage(rows[i], ths[k]) == pred[i], (i, ths[k])
    return pred, ths


def hybrid(rule_pred: np.ndarray, b_prob: np.ndarray) -> np.ndarray:
    """Rules give the reason; the stage CNN only splits Stage 3 from Stage 4."""
    out = rule_pred.copy()
    out[(rule_pred == "3") & (b_prob[:, CLASSES.index("4")] > b_prob[:, CLASSES.index("3")])] = "4"
    return out


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"bal_acc_4": bal(merge4(y), merge4(p)), "bal_acc_5": bal(y, p),
            "stage4_sensitivity": float((p[y == "4"] == "4").mean()) if (y == "4").any() else float("nan")}


def boot_samples(groups: np.ndarray, n: int = N_BOOT, seed: int = 0) -> list[np.ndarray]:
    """Cluster bootstrap: resample sources, keep every image of a drawn source."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = [np.where(groups == g)[0] for g in uniq]
    return [np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))]) for _ in range(n)]


def report(y: np.ndarray, preds: dict[str, np.ndarray], groups: np.ndarray, diffs: dict) -> dict:
    idx = boot_samples(groups)
    out: dict = {"n_images": len(y), "n_sources": len(np.unique(groups)),
                 "label_counts": {c: int((y == c).sum()) for c in CLASSES}, "methods": {}, "differences": {}}
    ci = lambda v: [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])]  # noqa: E731
    for m, p in preds.items():
        s = metrics(y, p)
        boots = [metrics(y[i], p[i]) for i in idx]
        out["methods"][m] = {k: round(v, 3) for k, v in s.items()}
        for k in ("bal_acc_4", "bal_acc_5"):
            out["methods"][m][k + "_ci95"] = ci([b[k] for b in boots])
        cols = [*CLASSES, "DTI", "belirsiz"]
        out["methods"][m]["confusion"] = {c: {k: int(((y == c) & (p == k)).sum()) for k in cols} for c in CLASSES}
    for name, (a, b, metric) in diffs.items():
        d = [metrics(y[i], preds[a][i])[metric] - metrics(y[i], preds[b][i])[metric] for i in idx]
        out["differences"][name] = {"metric": metric, "diff": round(metrics(y, preds[a])[metric]
                                                                    - metrics(y, preds[b])[metric], 3),
                                    "ci95": ci(d)}
    return out


def show(rep: dict) -> None:
    print(f"n = {rep['n_images']} görüntü / {rep['n_sources']} kaynak, {rep['label_counts']}")
    for m, s in rep["methods"].items():
        print(f"  {m:34s} 4-sınıf {s['bal_acc_4']:.3f} {s['bal_acc_4_ci95']}  5-sınıf {s['bal_acc_5']:.3f} "
              f"{s['bal_acc_5_ci95']}  Evre 4 duyarlılık {s['stage4_sensitivity']:.2f}")
    for k, v in rep["differences"].items():
        print(f"  {k}: {v}")


# ---------------------------------------------------------------- petchs OOF
def b_oof(arm: str) -> dict[str, np.ndarray]:
    """file -> seed-averaged held-out class probabilities."""
    out = {}
    for k in range(3):
        seeds = [json.loads((STEP3 / f"{arm}_fold{k}_s{s}.json").read_text())["pred"] for s in range(3)]
        out.update({n: np.mean([sd[n]["probs"] for sd in seeds], 0) for n in seeds[0]})
    return out


def petchs() -> dict:
    fd = json.loads(FEAT_D.read_text())
    fo = json.loads(FEAT_OLD.read_text())
    names = sorted(fd)
    assert names == sorted(fo)
    y = np.array([fd[n]["label"] for n in names], object)
    fold = np.array([fd[n]["fold"] for n in names])
    a_old, th_old = nested_rules([fo[n]["R2"] for n in names], y, fold)
    a_d, th_d = nested_rules([fd[n]["D"] for n in names], y, fold)
    preds = {"hep Evre 3": np.full(len(y), "3", object), "A × old-first": a_old, "A-D": a_d}
    probs = {}
    for arm, tag in (("ba", "B(a)"), ("bb", "B(b)")):
        b = b_oof(arm)
        probs[tag] = np.array([b[n] for n in names])
        preds[tag] = np.array([CLASSES[i] for i in probs[tag].argmax(1)], object)
    preds["H"] = hybrid(a_d, probs["B(a)"])
    preds["H-b"] = hybrid(a_d, probs["B(b)"])
    rep = report(y, preds, np.array(names), {
        "karar 1 (destekleyici): B(b) − B(a)": ("B(b)", "B(a)", "bal_acc_4"),
        "karar 2: A-D − A × old-first": ("A-D", "A × old-first", "bal_acc_4"),
        "karar 3 (petchs): H − A-D": ("H", "A-D", "bal_acc_5"),
        "ikincil: H-b − A-D": ("H-b", "A-D", "bal_acc_5"),
        "bilgi: B(a) − A-D": ("B(a)", "A-D", "bal_acc_4")})
    rep["fold_thresholds"] = {"A × old-first": th_old, "A-D": th_d}
    rep["per_image"] = {n: {"label": str(y[i]), **{m: str(p[i]) for m, p in preds.items()}}
                        for i, n in enumerate(names)}
    return rep


# ---------------------------------------------------------------- kaggle (locked external test)
@torch.no_grad()
def tissue_avg(cnns, img: np.ndarray, region: np.ndarray) -> np.ndarray | None:
    """Tissue label map inside region from the mean softmax of the three fold tissue CNNs."""
    from training.tissue_ab_compare import crop_resize, paste
    from training.train_tissue_cnn import NET_TO_AB, letterbox, to_tensor

    if region.sum() < 30:
        return None
    crop, reg, _, box = crop_resize(img, region)
    x, (nh, nw) = letterbox(crop, cv2.INTER_AREA)
    t = to_tensor(x)[None].to(next(cnns[0].parameters()).device)
    prob = torch.stack([torch.softmax(m(t)["out"][0, :, :nh, :nw], 0) for m in cnns]).mean(0)
    cls = cv2.resize(prob.argmax(0).cpu().numpy().astype(np.uint8), (crop.shape[1], crop.shape[0]),
                     interpolation=cv2.INTER_NEAREST)
    lbl = NET_TO_AB[cnns[0].n_cls][cls]
    lbl[~reg] = 0
    full = paste(lbl, box, img.shape[:2])
    full[~region] = 0
    return full


def kaggle() -> dict:
    from ultralytics import YOLO

    from sorbed.preprocess.color_norm import gray_world_normalize
    from training.eval_layered_map import deeplab_prob
    from training.tissue_ab_compare import SEG_NEW, SEG_OLD, wound_mask
    from training.train_stage_cnn import StageNet, device, predict
    from training.train_tissue_cnn import load as load_cnn
    from training.train_wound_deeplab import build_model
    from training.train_wound_deeplab import load as load_deeplab

    # every choice frozen on petchs: thresholds tuned on all 166 OOF feature rows (erythema 0.15)
    fd, fo = json.loads(FEAT_D.read_text()), json.loads(FEAT_OLD.read_text())
    names = sorted(fd)
    y_p = np.array([fd[n]["label"] for n in names], object)
    th_d = tune3(columns([fd[n]["D"] for n in names]), y_p)
    th_old = tune3(columns([fo[n]["R2"] for n in names]), y_p)

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

    rows = json.loads(LISTS.read_text())["kaggle"]
    feats_d, feats_old, probs = [], [], {"ba": [], "bb": []}
    for i, r in enumerate(rows):
        img = cv2.imread(r["file"])
        rgb = gray_world_normalize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        d = np.mean([deeplab_prob(m, img) for m in dls], 0) > 0.5
        old = wound_mask(segs, img, "old-first")
        feats_d.append(dataclasses.asdict(features(rgb, d, tissue_avg(cnns, img, d))))
        feats_old.append(dataclasses.asdict(features(rgb, old, tissue_avg(cnns, img, old))))
        for arm in probs:
            probs[arm].append(np.mean([predict(m, [img])[0] for m in b_models[arm]], 0))
        print(i + 1, len(rows), flush=True)
    probs = {a: np.array(v) for a, v in probs.items()}

    y = np.array([r["label"] for r in rows], object)
    groups = np.array([r["source"] for r in rows])
    preds = {"hep Evre 3": np.full(len(y), "3", object),
             "A × old-first": rules_fast(columns(feats_old), th_old), "A-D": rules_fast(columns(feats_d), th_d)}
    for arm, tag in (("ba", "B(a)"), ("bb", "B(b)")):
        preds[tag] = np.array([CLASSES[i] for i in probs[arm].argmax(1)], object)
    preds["H"] = hybrid(preds["A-D"], probs["ba"])
    preds["H-b"] = hybrid(preds["A-D"], probs["bb"])
    prim = y != "SDTI"
    rep = report(y[prim], {m: p[prim] for m, p in preds.items()}, groups[prim], {
        "karar 1 (birincil): B(b) − B(a)": ("B(b)", "B(a)", "bal_acc_4"),
        "karar 3 (kaggle): H − A-D": ("H", "A-D", "bal_acc_5"),
        "ikincil: H-b − A-D": ("H-b", "A-D", "bal_acc_5"),
        "bilgi: A-D − A × old-first": ("A-D", "A × old-first", "bal_acc_4")})
    rep["thresholds_all_petchs"] = {"A × old-first": th_old, "A-D": th_d}
    rep["SDTI"] = {m: {k: int((p[~prim] == k).sum()) for k in sorted(set(p[~prim]))} for m, p in preds.items()}
    rep["per_image"] = [{"file": r["file"], "label": r["label"], "source": r["source"],
                         **{m: str(p[i]) for m, p in preds.items()}} for i, r in enumerate(rows)]
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--petchs", action="store_true")
    ap.add_argument("--kaggle", action="store_true")
    args = ap.parse_args()
    if args.features:
        extract_d()
    for flag, fn, name in ((args.petchs, petchs, "petchs_oof.json"), (args.kaggle, kaggle, "kaggle.json")):
        if flag:
            if name == "kaggle.json" and (STEP3 / name).exists():
                raise SystemExit("kaggle was already measured once; the pre-registration allows a single run")
            rep = fn()
            (STEP3 / name).write_text(json.dumps(rep, ensure_ascii=False, indent=1))
            show(rep)


if __name__ == "__main__":
    main()
