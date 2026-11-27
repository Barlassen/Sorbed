"""Staging step 3, step 0: lock the kaggle external test set and build the fr7kn extra-source list.

Pre-registered in HANDOFF (2026-11-04). Two passes:

* ``--candidates`` hashes everything (dHash under the 8 rotations/flips), writes contact sheets of every kaggle image
  that is close to a petchs/fr7kn image or to another kaggle image into ``runs/audit/step3_dups/`` (patient photos:
  local only) and the candidate list into ``runs/staging/step3_candidates.json``.
* ``--lock`` reads the eye-review verdicts (``runs/staging/step3_dup_review.json``: {"confirmed": [pair ids]}),
  drops confirmed external duplicates from kaggle, merges confirmed kaggle-internal duplicates into one source and writes
  ``runs/staging/step3_lists.json``. Nothing here looks at a model output.

    python -m training.step3_lock --candidates
    python -m training.step3_lock --lock
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from training.petchs_groups import RF_PETCHS, SPLITS, source
from training.prepare_deeplab_folds import dhash_variants

FR7KN = Path("data/ruler_stage_fr7kn")
KAGGLE = Path("data/corpus/kaggle_stages/Dataset")
KAGGLE_LABELS = {"Stage_I": "1", "Stage_II": "2", "Stage_III": "3", "Stage_IV": "4", "Unstageable": "U",
                 "SDTI": "SDTI"}  # Invalid is dropped
CAND = Path("runs/staging/step3_candidates.json")
REVIEW = Path("runs/staging/step3_dup_review.json")
LISTS = Path("runs/staging/step3_lists.json")
SHEETS = Path("runs/audit/step3_dups")
EXT_HAMMING = 8  # kaggle vs petchs/fr7kn candidates (confirmed by eye)
INT_HAMMING = 6  # kaggle-internal candidates (confirmed by eye); fr7kn extra vs petchs (automatic)


def kaggle_files() -> list[tuple[Path, str]]:
    return [(p, lab) for d, lab in KAGGLE_LABELS.items() for p in sorted((KAGGLE / d).iterdir())
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]


def fr7kn_labels() -> dict[str, set[str]]:
    """file name -> stage labels from the train/test class folders."""
    out: dict[str, set[str]] = defaultdict(set)
    for sp in ("train", "test"):
        for c in "1234":
            for p in (FR7KN / sp / c).iterdir():
                out[p.name].add(c)
    return out


def petchs_files() -> list[Path]:
    return [p for sp in SPLITS for p in sorted((RF_PETCHS / sp).iterdir()) if p.suffix.lower() in {".jpg", ".png"}]


def hashes(paths: list[Path]) -> np.ndarray:
    """(n, 8) uint64 dHashes under the 8 rotations/flips."""
    return np.array([dhash_variants(cv2.imread(str(p))) for p in paths], dtype=np.uint64)


def min_hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(len(a), len(b)) minimum Hamming distance between any variant of a and the identity hash of b."""
    return np.bitwise_count(a[:, :, None] ^ b[None, None, :, 0]).min(1)


def sheet(pairs: list[tuple[str, Path, Path, int]], out: Path) -> None:
    h = 260
    rows = []
    for pid, a, b, d in pairs:
        ims = []
        for p in (a, b):
            im = cv2.imread(str(p))
            ims.append(cv2.resize(im, (round(im.shape[1] * h / im.shape[0]), h), interpolation=cv2.INTER_AREA))
        row = np.hstack([ims[0], np.full((h, 12, 3), 255, np.uint8), ims[1]])
        cv2.putText(row, f"{pid}  d={d}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        rows.append(row)
    w = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.zeros((h, w - r.shape[1], 3), np.uint8)]) for r in rows]
    cv2.imwrite(str(out), np.vstack(rows))


def candidates() -> None:
    kag = kaggle_files()
    pet = petchs_files()
    fr_lab = fr7kn_labels()
    fr = sorted(FR7KN / "all_images" / n for n in fr_lab)
    print(f"kaggle {len(kag)}, petchs {len(pet)}, fr7kn {len(fr)}; hashing ...", flush=True)
    hk, hp, hf = hashes([p for p, _ in kag]), hashes(pet), hashes(fr)

    ext = []
    for other, ho in (("petchs", hp), ("fr7kn", hf)):
        paths = pet if other == "petchs" else fr
        d = min_hamming(hk, ho)
        for i, j in np.argwhere(d <= EXT_HAMMING):
            ext.append({"id": f"x{len(ext):03d}", "kaggle": str(kag[i][0]), "other": str(paths[j]),
                        "set": other, "hamming": int(d[i, j])})
    d = min_hamming(hk, hk)
    internal = [{"id": f"k{n:03d}", "a": str(kag[i][0]), "b": str(kag[j][0]), "hamming": int(d[i, j])}
                for n, (i, j) in enumerate((i, j) for i, j in np.argwhere(d <= INT_HAMMING) if i < j)]

    # fr7kn extra sources: not a petchs source, a single stage label, no rotated/flipped copy of a petchs image
    pet_src = {source(p.name) for p in pet}
    by_src: dict[str, list[int]] = defaultdict(list)
    for j, p in enumerate(fr):
        by_src[source(p.name)].append(j)
    dfp = min_hamming(hp, hf).min(0)  # per fr7kn image: closest petchs image
    extra, multi, matched = {}, [], []
    for s, idx in sorted(by_src.items()):
        if s in pet_src:
            continue
        labels = set().union(*(fr_lab[fr[j].name] for j in idx))
        if len(labels) > 1:
            multi.append(s)
        elif (dfp[idx] <= INT_HAMMING).any():
            matched.append(s)
        else:
            extra[s] = {"label": labels.pop(), "files": [fr[j].name for j in idx]}

    CAND.parent.mkdir(parents=True, exist_ok=True)
    CAND.write_text(json.dumps({"external": ext, "internal": internal, "fr7kn_extra": extra,
                                "fr7kn_dropped_multilabel": multi, "fr7kn_dropped_petchs_match": matched}, indent=1))
    SHEETS.mkdir(parents=True, exist_ok=True)
    for tag, pairs in (("external", [(c["id"], Path(c["kaggle"]), Path(c["other"]), c["hamming"]) for c in ext]),
                       ("internal", [(c["id"], Path(c["a"]), Path(c["b"]), c["hamming"]) for c in internal])):
        for n in range(0, len(pairs), 8):
            sheet(pairs[n:n + 8], SHEETS / f"{tag}_{n // 8:02d}.png")
    print(json.dumps({"external_candidates": len(ext), "internal_pairs": len(internal),
                      "fr7kn_sources": len(by_src), "fr7kn_petchs_sources": len(set(by_src) & pet_src),
                      "fr7kn_extra": len(extra), "extra_images": sum(len(v["files"]) for v in extra.values()),
                      "dropped_multilabel": len(multi), "dropped_petchs_match": len(matched)}, indent=1))


def lock() -> None:
    cand = json.loads(CAND.read_text())
    confirmed = set(json.loads(REVIEW.read_text())["confirmed"])
    drop = {c["kaggle"] for c in cand["external"] if c["id"] in confirmed}
    kag = [(str(p), lab) for p, lab in kaggle_files()]
    parent = {p: p for p, _ in kag}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c in cand["internal"]:
        if c["id"] in confirmed:
            parent[find(c["a"])] = find(c["b"])
    rows = [{"file": p, "label": lab, "source": Path(find(p)).stem} for p, lab in kag if p not in drop]
    LISTS.write_text(json.dumps({
        "kaggle": rows, "kaggle_dropped_external_dups": sorted(drop),
        "fr7kn_extra": cand["fr7kn_extra"], "fr7kn_dropped_multilabel": cand["fr7kn_dropped_multilabel"],
        "fr7kn_dropped_petchs_match": cand["fr7kn_dropped_petchs_match"]}, indent=1))
    main_rows = [r for r in rows if r["label"] != "SDTI"]
    print(json.dumps({"kaggle_kept": len(rows), "dropped_external": len(drop),
                      "primary_images": len(main_rows), "primary_sources": len({r["source"] for r in main_rows}),
                      "sdti": len(rows) - len(main_rows), "fr7kn_extra_sources": len(cand["fr7kn_extra"])}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", action="store_true")
    ap.add_argument("--lock", action="store_true")
    args = ap.parse_args()
    if args.candidates:
        candidates()
    if args.lock:
        lock()


if __name__ == "__main__":
    main()
