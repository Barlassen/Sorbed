"""Staging step 4: lock the PIID pressure-injury set as a clean external test set.

Same recipe as ``training.step3_lock`` (kaggle). Two passes:

* ``--candidates`` hashes everything (dHash under the 8 rotations/flips) and lists every PIID image that is close to a
  petchs, fr7kn or kaggle image (external, Hamming <= 8) or to another PIID image (internal, Hamming <= 6). Writes the
  list to ``runs/staging/step4_candidates.json`` and contact sheets to ``runs/audit/step4_dups/`` (patient photos:
  local only). Sheets show only the pair id and the distance, never a stage label, so the review is blind to labels.
* ``--train-sources`` adds one more external set: every image the mask/tissue models trained on (DeepLab folds, YOLO
  segmenters, tissue CNN) that petchs/fr7kn do not already cover. Its ``xt`` pairs are appended to the candidate list
  (existing ids untouched) with sheets ``trainsrc_NN.png``.
* ``--lock`` also reads ``runs/staging/step4_crop_candidates.json`` (``training.step4_crop_dups``) if present, then
  reads the eye-review verdicts (``runs/staging/step4_dup_review.json``: {"confirmed": [pair ids]}),
  drops PIID images in confirmed external pairs, merges confirmed internal pairs into one source and writes
  ``runs/staging/step4_lists.json``. Nothing here looks at a model output.

    python -m training.step4_lock --candidates
    python -m training.step4_lock --train-sources
    python -m training.step4_lock --lock
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from training.step3_lock import FR7KN, fr7kn_labels, hashes, kaggle_files, min_hamming, petchs_files

PIID = Path("data/corpus/piid_data/dataset")
PIID_LABELS = ("1", "2", "3", "4")  # folder name = stage label
CAND = Path("runs/staging/step4_candidates.json")
CROP = Path("runs/staging/step4_crop_candidates.json")  # training.step4_crop_dups (SIFT crop/rescale pass)
REVIEW = Path("runs/staging/step4_dup_review.json")
LISTS = Path("runs/staging/step4_lists.json")
SHEETS = Path("runs/audit/step4_dups")
EXT_HAMMING = 8  # PIID vs petchs/fr7kn/kaggle candidates (confirmed by eye)
INT_HAMMING = 6  # PIID-internal candidates (confirmed by eye)
PER_SHEET = 6
ROW_H = 240
SET_TAG = {"petchs": "xp", "fr7kn": "xf", "kaggle": "xk", "trainsrc": "xt"}
IMG_EXT = {".png", ".jpg", ".jpeg"}
TRAIN_DIRS = (  # image folders the mask/tissue models trained on; YOLO folders hold symlinks plus __crop copies
    *(Path(f"data/yolo_seg_v5_fold{k}/images/{s}") for k in range(3) for s in ("train", "val")),
    *(Path(f"data/{d}/images/{s}") for d in ("yolo_seg_combined", "yolo_fuseg") for s in ("train", "val")),
    # Labeled/Padded is the same photos on black borders, used by no trainer; its near-empty dHashes collide widely
    Path("data/corpus/dfutissue/DFUTissue/Labeled/Original/Images"),
    Path("data/corpus/lutseg/Images"), Path("data/corpus/woundtissue/image"), Path("data/corpus/figshare_pu2"),
    Path("data/corpus/rf_segulcer/train"), Path("data/corpus/rf_segulcer/valid"))
TRAIN_COVERED = (Path("data/corpus/rf_petchs"), Path("data/corpus/roboflow_fr7kn"), FR7KN)  # already compared


def piid_files() -> list[tuple[Path, str]]:
    return [(p, lab) for lab in PIID_LABELS for p in sorted((PIID / lab).iterdir())
            if p.suffix.lower() in IMG_EXT]


def train_source_files() -> dict[str, str]:
    """Real path (relative to the repo) -> source corpus, for every training image not covered by another pass.

    Symlinks are resolved and deduped by real path; ``__crop`` copies and label/mask images are skipped.
    """
    root = Path.cwd().resolve()
    covered = [c.resolve() for c in TRAIN_COVERED]
    out: dict[str, str] = {}
    for d in TRAIN_DIRS:
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() not in IMG_EXT or "__crop" in p.name:
                continue
            r = p.resolve()
            rel = r.relative_to(root).parts
            if any("mask" in s or s in {"label", "labels", "annotations", "palette"} for s in map(str.lower, rel)):
                continue
            if any(r.is_relative_to(c) for c in covered):
                continue
            out.setdefault(str(Path(*rel)), rel[2] if rel[1] == "corpus" else rel[1])
    return out


def sheet(pairs: list[tuple[str, Path, Path, int]], out: Path) -> None:
    """One row per pair: the two images side by side, with only the pair id and the Hamming distance printed."""
    rows = []
    for pid, a, b, d in pairs:
        ims = []
        for p in (a, b):
            im = cv2.imread(str(p))
            ims.append(cv2.resize(im, (max(1, round(im.shape[1] * ROW_H / im.shape[0])), ROW_H),
                                  interpolation=cv2.INTER_AREA))
        row = np.hstack([ims[0], np.full((ROW_H, 12, 3), 255, np.uint8), ims[1]])
        txt = f"{pid}  d={d}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(row, (0, 0), (tw + 12, th + 16), (0, 0, 0), -1)
        cv2.putText(row, txt, (6, th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        rows.append(row)
    w = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.zeros((ROW_H, w - r.shape[1], 3), np.uint8)]) for r in rows]
    sep = np.full((6, w, 3), 128, np.uint8)
    cv2.imwrite(str(out), np.vstack([x for r in rows for x in (r, sep)][:-1]))


def candidates() -> None:
    piid = [p for p, _ in piid_files()]
    pet = petchs_files()
    fr = sorted(FR7KN / "all_images" / n for n in fr7kn_labels())
    kag = [p for p, _ in kaggle_files()]
    print(f"piid {len(piid)}, petchs {len(pet)}, fr7kn {len(fr)}, kaggle {len(kag)}; hashing ...", flush=True)
    hi, hp, hf, hk = hashes(piid), hashes(pet), hashes(fr), hashes(kag)

    ext: dict[str, list[dict]] = {}
    for other, paths, ho in (("petchs", pet, hp), ("fr7kn", fr, hf), ("kaggle", kag, hk)):
        d = min_hamming(hi, ho)
        found = sorted((int(d[i, j]), str(piid[i]), str(paths[j])) for i, j in np.argwhere(d <= EXT_HAMMING))
        ext[other] = [{"id": f"{SET_TAG[other]}{n:03d}", "piid": a, "other": b, "set": other, "hamming": h}
                      for n, (h, a, b) in enumerate(found)]
    d = min_hamming(hi, hi)
    d = np.minimum(d, d.T)  # symmetric: a rotated/flipped copy can be caught from either side
    found = sorted((int(d[i, j]), str(piid[i]), str(piid[j])) for i, j in np.argwhere(d <= INT_HAMMING) if i < j)
    internal = [{"id": f"i{n:03d}", "piid": a, "other": b, "set": "internal", "hamming": h}
                for n, (h, a, b) in enumerate(found)]

    CAND.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(CAND.read_text()) if CAND.exists() else {}  # keep the --train-sources pass's entries
    CAND.write_text(json.dumps({
        "external": [c for v in ext.values() for c in v] + [c for c in old.get("external", []) if c["set"] == "trainsrc"],
        "internal": internal, **{k: v for k, v in old.items() if k == "trainsrc_hashed"}}, indent=1))
    SHEETS.mkdir(parents=True, exist_ok=True)
    for old in SHEETS.glob("*.png"):
        if not old.name.startswith(("trainsrc_", "crop_")):  # other passes' sheets stay
            old.unlink()
    groups = (*ext.items(), ("internal", internal))
    sheets = {k: v for tag, cands in groups for k, v in write_sheets(tag, cands).items()}
    print(json.dumps({"summary": {tag: summarize(tag, cands) for tag, cands in groups}, "sheets": sheets}, indent=1))


def write_sheets(tag: str, cands: list[dict]) -> dict[str, str]:
    """Sheets ``{tag}_NN.png``, PER_SHEET pairs each; returns sheet name -> id range."""
    pairs = [(c["id"], Path(c["piid"]), Path(c["other"]), c["hamming"]) for c in cands]
    out = {}
    for n in range(0, len(pairs), PER_SHEET):
        f = SHEETS / f"{tag}_{n // PER_SHEET:02d}.png"
        sheet(pairs[n:n + PER_SHEET], f)
        out[f.name] = f"{pairs[n][0]}-{pairs[min(n + PER_SHEET, len(pairs)) - 1][0]}"
    return out


def summarize(tag: str, cands: list[dict]) -> dict:
    hd = Counter(c["hamming"] for c in cands)
    imgs = {c["piid"] for c in cands} | ({c["other"] for c in cands} if tag == "internal" else set())
    return {"pairs": len(cands), "piid_images": len(imgs), "other_images": len({c["other"] for c in cands}),
            "hamming_0": hd.get(0, 0), "hamming_hist": dict(sorted(hd.items()))}


def train_sources() -> None:
    """PIID vs every training image no other pass covers; appends ``xt`` pairs, leaves existing candidates alone."""
    cand = json.loads(CAND.read_text())
    piid = [p for p, _ in piid_files()]
    src = train_source_files()
    files = sorted(src)
    counts = dict(sorted(Counter(src.values()).items()))
    print(f"piid {len(piid)}, training images {len(files)} {counts}; hashing ...", flush=True)
    d = min_hamming(hashes(piid), hashes([Path(f) for f in files]))
    found = sorted((int(d[i, j]), str(piid[i]), files[j]) for i, j in np.argwhere(d <= EXT_HAMMING))
    new = [{"id": f"xt{n:03d}", "piid": a, "other": b, "set": "trainsrc", "hamming": h}
           for n, (h, a, b) in enumerate(found)]
    cand["external"] = [c for c in cand["external"] if c["set"] != "trainsrc"] + new
    cand["trainsrc_hashed"] = counts
    CAND.write_text(json.dumps(cand, indent=1))
    for old in SHEETS.glob("trainsrc_*.png"):
        old.unlink()
    SHEETS.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"summary": summarize("trainsrc", new), "sheets": write_sheets("trainsrc", new)}, indent=1))


def lock() -> None:
    cand = json.loads(CAND.read_text())
    if CROP.exists():  # crop pass: cf/ck/cp external, ci internal; near-threshold rows reviewed as nt<index>
        crop = json.loads(CROP.read_text())
        nt = [{"id": f"nt{i}", "piid": r["query"], "other": r["other"], "set": r["set"]}
              for i, r in enumerate(crop["near_threshold"]) if not r["set"].startswith("kk_")]  # kk = kaggle check
        for c in crop["piid"] + nt:
            cand["internal" if c["set"] == "internal" else "external"].append(c)
    confirmed = set(json.loads(REVIEW.read_text())["confirmed"])
    known = {c["id"] for c in cand["external"] + cand["internal"]}
    if unknown := sorted(confirmed - known):
        raise SystemExit(f"unknown pair ids in {REVIEW}: {unknown}")
    drop = {c["piid"] for c in cand["external"] if c["id"] in confirmed}
    piid = [(str(p), lab) for p, lab in piid_files()]
    parent = {p: p for p, _ in piid}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c in cand["internal"]:
        if c["id"] in confirmed:
            parent[find(c["piid"])] = find(c["other"])
    rows = [{"file": p, "label": lab, "source": Path(find(p)).stem} for p, lab in piid if p not in drop]
    LISTS.parent.mkdir(parents=True, exist_ok=True)
    LISTS.write_text(json.dumps({"piid": rows, "piid_dropped_external_dups": sorted(drop)}, indent=1))
    src_labels: dict[str, set[str]] = {}
    for r in rows:
        src_labels.setdefault(r["source"], set()).add(r["label"])
    print(json.dumps({"piid_kept": len(rows), "piid_sources": len(src_labels),
                      "per_label": dict(sorted(Counter(r["label"] for r in rows).items())),
                      "dropped_external": len(drop),
                      "mixed_label_sources": sum(len(v) > 1 for v in src_labels.values())}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", action="store_true")
    ap.add_argument("--train-sources", action="store_true")
    ap.add_argument("--lock", action="store_true")
    args = ap.parse_args()
    if args.candidates:
        candidates()
    if args.train_sources:
        train_sources()
    if args.lock:
        lock()


if __name__ == "__main__":
    main()
