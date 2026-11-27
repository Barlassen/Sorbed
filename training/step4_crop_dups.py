"""Staging step 4: find crop / rescale duplicates that the dHash search misses.

dHash (``training.step4_lock``, ``training.step3_lock``) compares whole-image thumbnails, so a crop or a stretched
rescale of a photo does not match it (known case: PIID ``1/stage_1_49681.jpg`` is a ~60 % wide crop of kaggle
``Stage_I/Stage_I_014.png``). This script uses one local feature (SIFT, cv2; nothing downloaded, no network model):

1. Features. SIFT on the gray image, long side 384 px, very low contrast threshold (skin photos are smooth, the
   default finds almost no keypoints), strongest ``SIFT_N`` kept. Query images (PIID, kaggle test) are also described
   mirrored; SIFT itself is invariant to crop, scale and rotation.
2. Retrieval. A FLANN kd-tree over the strongest ``VOTE_N`` descriptors of every image of a set; each descriptor of a
   query votes for the image of its nearest neighbour (plain and mirrored query, best of the two). Each query keeps
   its top ``TOP_K`` images of every set. This finds the pair whichever side is the crop. (A COCO-DeepLab CNN
   embedding was tried first: on synthetic crops it put the source in the top 10 for only 68 % of fr7kn images.)
3. Verification. Lowe ratio test between the two images' descriptors and a RANSAC affine fit (crop, rescale,
   stretch, rotation); score = RANSAC inliers, best of plain/mirrored query. Candidate: score >= ``MIN_INLIERS``,
   checked against the known pair, synthetic crops and random unrelated pairs (stored under "calibration"). Each
   candidate also gets "ncc", the pixel correlation of the aligned overlap (~1 = same photograph; low = the inliers
   sit on a shared overlay or the photos differ). It is reported for the reviewers, not used as a gate.

Query -> reference sets and pair ids:
* PIID vs petchs (cp###), all fr7kn (cf###), kaggle (ck###) and itself (ci###). Pairs already in
  ``runs/staging/step4_candidates.json`` (dHash) are left out; for petchs/fr7kn a pair is (PIID image, roboflow
  source), so one row per source and augmented copies of an already listed source are not shown again.
* kaggle test images (``step3_lists.json``) vs petchs and the fr7kn extra training images (kk###): a post-hoc leak
  check of the test that was already run; it changes nothing. Pairs that step3's dHash review already saw carry that
  id under "step3".

Pairs scoring NEAR..MIN_INLIERS-1 (above every random pair, below the threshold) are listed under "near_threshold"
in the JSON but not put on sheets.

Writes ``runs/staging/step4_crop_candidates.json`` and contact sheets ``runs/audit/step4_dups/crop_NN.png`` (6 pairs
per sheet, only "id s=score" printed, no labels). Patient photos: local only.

    PYTHONPATH=. .venv/bin/python -m training.step4_crop_dups
"""

from __future__ import annotations

import json
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from training.petchs_groups import source
from training.step3_lock import CAND as STEP3_CAND
from training.step3_lock import FR7KN, fr7kn_labels, kaggle_files, petchs_files
from training.step3_lock import LISTS as STEP3_LISTS
from training.step4_lock import CAND, PER_SHEET, ROW_H, SHEETS, piid_files

OUT = Path("runs/staging/step4_crop_candidates.json")
CONTROL = ("data/corpus/piid_data/dataset/1/stage_1_49681.jpg", "data/corpus/kaggle_stages/Dataset/Stage_I/Stage_I_014.png")
CONFIRMED = ("i000", "i001", "i002", "i003", "i004", "i026", "xk000")  # dHash pairs confirmed by eye so far
SIFT_SIDE = 384  # long side for SIFT
SIFT_N = 1500
SIFT_CONTRAST = 0.002  # default 0.04 finds ~0 keypoints on smooth skin photos
VOTE_N = 1000  # strongest descriptors per image used for retrieval votes
CHECKS = 64  # FLANN search effort
TOP_K = 10
RATIO = 0.8
RANSAC_PX = 5.0
MIN_INLIERS = 20
NEAR = 12  # random unrelated pairs never reached 12 (max 11 of 3000)
TAG = {"petchs": "cp", "fr7kn": "cf", "kaggle": "ck", "internal": "ci"}

Feat = tuple[np.ndarray, np.ndarray]  # keypoint xy (n, 2) float32, descriptors (n, 128) uint8; strongest first
_tls = threading.local()


def _cv() -> threading.local:
    if not hasattr(_tls, "sift"):
        _tls.sift = cv2.SIFT_create(nfeatures=SIFT_N, contrastThreshold=SIFT_CONTRAST)
    return _tls


# ----------------------------------------------------------------- features
def gray(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = SIFT_SIDE / max(g.shape)
    return cv2.resize(g, (round(g.shape[1] * s), round(g.shape[0] * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def keypoints(img: np.ndarray, mirror: bool = False) -> Feat:
    g = gray(img)
    if mirror:
        g = np.ascontiguousarray(g[:, ::-1])
    kp, d = _cv().sift.detectAndCompute(g, None)
    if d is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8)
    order = np.argsort([-k.response for k in kp], kind="stable")
    return np.float32([kp[i].pt for i in order]).reshape(-1, 2), d[order].astype(np.uint8)  # SIFT values are integers


class Features:
    """SIFT features per file, plain and (for query files) mirrored."""

    def __init__(self, pool: ThreadPoolExecutor) -> None:
        self.pool = pool
        self.f: dict[tuple[str, bool], Feat] = {}

    def add(self, files: list[str], mirror: bool = False) -> None:
        todo = [(f, m) for f in files for m in ((False, True) if mirror else (False,)) if (f, m) not in self.f]
        for k, ft in zip(todo, self.pool.map(lambda t: keypoints(cv2.imread(t[0]), t[1]), todo), strict=True):
            self.f[k] = ft

    def __getitem__(self, k: tuple[str, bool]) -> Feat:
        return self.f[k]


# ---------------------------------------------------------------- retrieval
class VoteIndex:
    """FLANN kd-tree over the strongest descriptors of a list of images (owner = image index)."""

    def __init__(self, descs: list[np.ndarray], owners: list[int]) -> None:
        d = [x[:VOTE_N] for x in descs]
        self.owner = np.concatenate([np.full(len(x), o) for x, o in zip(d, owners, strict=True)])
        self.n = max(owners) + 1
        self.index = cv2.flann.Index(np.concatenate(d).astype(np.float32), {"algorithm": 1, "trees": 4})

    def votes(self, descs: list[np.ndarray], skip: int = -1) -> np.ndarray:
        """Votes per image (best over the given descriptor sets); nearest neighbours owned by ``skip`` are passed over."""
        best = np.zeros(self.n)
        for d in descs:
            if len(d) == 0:
                continue
            nn, _ = self.index.knnSearch(np.ascontiguousarray(d[:VOTE_N], np.float32), 2, params={"checks": CHECKS})
            own = self.owner[nn]
            pick = np.where(own[:, 0] == skip, own[:, 1], own[:, 0])
            pick = pick[pick != skip]
            best = np.maximum(best, np.bincount(pick, minlength=self.n))
        return best


def retrieve(qs: list[str], rs: list[str], F: Features, pool: ThreadPoolExecutor, internal: bool = False,
             ) -> tuple[set[tuple[int, int]], VoteIndex, np.ndarray]:
    """Top TOP_K references per query by SIFT votes (i < j pairs when qs is rs); also returns the reference index and
    the (nq, nr) vote matrix for calibration."""
    index = VoteIndex([F[(r, False)][1] for r in rs], list(range(len(rs))))
    votes = np.stack(list(pool.map(lambda i: index.votes([F[(qs[i], False)][1], F[(qs[i], True)][1]],
                                                         skip=i if internal else -1), range(len(qs)))))
    pairs = {(i, int(j)) for i, row in enumerate(votes) for j in np.argsort(-row, kind="stable")[:TOP_K] if row[j] > 0}
    if internal:
        pairs = {(min(i, j), max(i, j)) for i, j in pairs}
    return pairs, index, votes


# ------------------------------------------------------------- verification
def fit(fa: Feat, fb: Feat) -> tuple[int, np.ndarray | None]:
    """RANSAC affine fit (a -> b) to the ratio-test SIFT matches: (inliers, 2x3 matrix)."""
    (pa, da), (pb, db) = fa, fb
    if len(da) < 3 or len(db) < 3:
        return 0, None
    a, b = da.astype(np.float32), db.astype(np.float32)
    d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T  # squared distances (same result as BFMatcher)
    nn = np.argpartition(d2, 1, axis=1)[:, :2]
    v = np.sqrt(np.maximum(np.take_along_axis(d2, nn, 1), 0))
    j = np.where(v[:, 0] <= v[:, 1], nn[:, 0], nn[:, 1])
    good = np.nonzero(v.min(1) < RATIO * v.max(1))[0]
    if len(good) < 4:
        return 0, None
    M, inl = cv2.estimateAffine2D(pa[good], pb[j[good]], method=cv2.RANSAC,
                                  ransacReprojThreshold=RANSAC_PX, maxIters=2000, confidence=0.995)
    if M is None:
        return 0, None
    sv = np.linalg.svd(M[:, :2], compute_uv=False)
    if sv[1] < 0.1 or sv[0] > 10:  # degenerate fit
        return 0, None
    return int(inl.sum()), M


def match(fa: Feat, fb: Feat) -> int:
    return fit(fa, fb)[0]


def score(F: Features, a: str, b: str) -> int:
    """Best of plain and mirrored a against b."""
    fb = F[(b, False)]
    return max(match(F[(a, False)], fb), match(F[(a, True)], fb))


def overlap_ncc(F: Features, a: str, b: str) -> float:
    """Pixel correlation of a with b warped onto it by the best fit (gray, blurred, overlap only).

    Close to 1 when the overlap is the same photograph; low when the inliers sit on a shared overlay (text, logo,
    ruler) but the pictures differ. Reported next to the score, not used as a gate."""
    fits = [(*fit(F[(a, m)], F[(b, False)]), m) for m in (False, True)]
    _, M, mirror = max(fits, key=lambda t: t[0])
    if M is None:
        return 0.0
    ga, gb = gray(cv2.imread(a)), gray(cv2.imread(b))
    if mirror:
        ga = np.ascontiguousarray(ga[:, ::-1])
    h, w = ga.shape
    wb = cv2.warpAffine(gb, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    ok = cv2.warpAffine(np.full_like(gb, 255), M, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    ok = cv2.erode(ok, np.ones((5, 5), np.uint8)) > 0
    if ok.sum() < 500:
        return 0.0
    x, y = (cv2.GaussianBlur(g.astype(np.float32), (0, 0), 2)[ok] for g in (ga, wb))
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else 0.0


# -------------------------------------------------------------- calibration
def synthetic_crop(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Random crop (45-90 % of each side), resize/stretch, rotation/flip, brightness and JPEG."""
    h, w = img.shape[:2]
    fw, fh = rng.uniform(0.45, 0.9), rng.uniform(0.45, 0.9)
    x0, y0 = rng.uniform(0, 1 - fw), rng.uniform(0, 1 - fh)
    c = img[int(y0 * h):int((y0 + fh) * h), int(x0 * w):int((x0 + fw) * w)]
    c = cv2.resize(c, rng.choice([(224, 224), (300, 225), (256, 192)]), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        c = c[:, ::-1]
    c = np.ascontiguousarray(np.rot90(c, rng.randrange(4)))
    c = np.clip(c.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15), 0, 255).astype(np.uint8)
    return cv2.imdecode(cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 75])[1], cv2.IMREAD_COLOR)


def pct(x: list[int] | np.ndarray) -> dict[str, float]:
    x = np.asarray(x)
    return {"n": len(x), **{f"p{q}": float(np.percentile(x, q)) for q in (1, 5, 10, 50, 90, 99, 99.9)},
            "min": float(x.min()), "max": float(x.max())}


def key(name: str, q: str, o: str) -> tuple:
    """Pair identity: (set, query, roboflow source) for petchs/fr7kn, unordered files for internal, else files."""
    if name in ("petchs", "fr7kn"):
        return (name, q, source(Path(o).name))
    if name == "internal":
        return ("internal", *sorted((q, o)))
    return (name, q, o)


# ------------------------------------------------------------------- sheets
def sheet(pairs: list[tuple[str, str, str, int]], out: Path) -> None:
    rows = []
    for pid, a, b, s in pairs:
        ims = []
        for p in (a, b):
            im = cv2.imread(p)
            ims.append(cv2.resize(im, (max(1, round(im.shape[1] * ROW_H / im.shape[0])), ROW_H), interpolation=cv2.INTER_AREA))
        row = np.hstack([ims[0], np.full((ROW_H, 12, 3), 255, np.uint8), ims[1]])
        txt = f"{pid}  s={s}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(row, (0, 0), (tw + 12, th + 16), (0, 0, 0), -1)
        cv2.putText(row, txt, (6, th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        rows.append(row)
    w = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.zeros((ROW_H, w - r.shape[1], 3), np.uint8)]) for r in rows]
    sep = np.full((6, w, 3), 128, np.uint8)
    cv2.imwrite(str(out), np.vstack([x for r in rows for x in (r, sep)][:-1]))


# --------------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    pool = ThreadPoolExecutor(8)
    piid = [str(p) for p, _ in piid_files()]
    sets = {"petchs": [str(p) for p in petchs_files()],
            "fr7kn": [str(FR7KN / "all_images" / n) for n in sorted(fr7kn_labels())],
            "kaggle": [str(p) for p, _ in kaggle_files()]}
    lists = json.loads(STEP3_LISTS.read_text())
    kag_test = [r["file"] for r in lists["kaggle"]]
    fr_extra = sorted(str(FR7KN / "all_images" / f) for v in lists["fr7kn_extra"].values() for f in v["files"])
    assert set(kag_test) <= set(sets["kaggle"]) and set(fr_extra) <= set(sets["fr7kn"])
    print(f"piid {len(piid)}, petchs {len(sets['petchs'])}, fr7kn {len(sets['fr7kn'])}, kaggle {len(sets['kaggle'])}, "
          f"kaggle test {len(kag_test)}, fr7kn extra {len(fr_extra)}", flush=True)
    F = Features(pool)
    F.add(sorted(set().union(*sets.values()) - set(kag_test)))
    F.add(piid + kag_test, mirror=True)
    t_feat = time.time() - t0
    print(f"SIFT features in {t_feat:.0f}s; retrieving ...", flush=True)

    dh = json.loads(CAND.read_text())
    dh_rows = [c for c in dh["external"] + dh["internal"] if c["set"] in TAG]  # the sets this script covers
    dh_key = {key(c["set"], c["piid"], c["other"]): c["id"] for c in dh_rows}
    rng = random.Random(0)

    # retrieval (and, while each reference index exists, the synthetic-crop retrieval check)
    runs = [("petchs", piid, sets["petchs"]), ("fr7kn", piid, sets["fr7kn"]), ("kaggle", piid, sets["kaggle"]),
            ("internal", piid, piid), ("kk_petchs", kag_test, sets["petchs"]), ("kk_fr7kn", kag_test, fr_extra)]
    retrieved: list[tuple[str, str, str]] = []
    syn: list[tuple[str, int, Feat, Feat]] = []  # (source file, retrieval rank, crop features plain / mirrored)
    control: dict = {"pair": list(CONTROL)}
    for name, qs, rs in runs:
        t = time.time()
        pairs, index, fwd = retrieve(qs, rs, F, pool, internal=name == "internal")
        retrieved += [(name, qs[i], rs[j]) for i, j in sorted(pairs)]
        if name == "kaggle":
            a, b = qs.index(CONTROL[0]), rs.index(CONTROL[1])
            control.update(votes=int(fwd[a, b]), rank_of_kaggle_for_piid=int((fwd[a] > fwd[a, b]).sum()) + 1,
                           retrieved=(a, b) in pairs)
        if name in sets:
            src = rng.sample(rs, 60)
            crops = [synthetic_crop(cv2.imread(f), rng) for f in src]
            cf = list(pool.map(lambda c: (keypoints(c), keypoints(c, True)), crops))
            srcs = np.array([source(Path(f).name) for f in rs])
            for f, (fp, fm) in zip(src, cf, strict=True):
                v = index.votes([fp[1], fm[1]])
                hit = v[rs.index(f)]
                rivals = srcs != source(Path(f).name)  # augmented copies of the same photo do not compete
                syn.append((f, int((v[rivals] >= hit).sum()) + 1, fp, fm))
        print(f"  {name}: {len(pairs)} pairs ({time.time() - t:.0f}s)", flush=True)
    t_ret = time.time() - t0 - t_feat
    print(f"{len(retrieved)} retrieved pairs; verifying ...", flush=True)

    scores = list(pool.map(lambda p: score(F, p[1], p[2]), retrieved))
    t_ver = time.time() - t0 - t_feat - t_ret

    # calibration
    control["score"] = score(F, *CONTROL)
    control["ncc"] = round(overlap_ncc(F, *CONTROL), 3)
    syn_rank = np.array([r for _, r, _, _ in syn])
    syn_score = np.array(list(pool.map(lambda s: max(match(s[2], F[(s[0], False)]), match(s[3], F[(s[0], False)])), syn)))
    refs = [(name, f) for name, fs in sets.items() for f in fs]
    rnd: list[tuple[str, str]] = []
    while len(rnd) < 3000:  # random unrelated pairs; pairs dHash listed are skipped
        q, (name, o) = rng.choice(piid), rng.choice(refs)
        if key(name, q, o) not in dh_key:
            rnd.append((q, o))
    rnd_score = np.array(list(pool.map(lambda p: score(F, *p), rnd)))
    calib = {
        "control": control,
        "synthetic_crops": {"n": len(syn), "retrieved_at_top_k": float(np.mean(syn_rank <= TOP_K)),
                            "score": pct(syn_score), "pass": float(np.mean(syn_score >= MIN_INLIERS)),
                            "retrieved_and_pass": float(np.mean((syn_rank <= TOP_K) & (syn_score >= MIN_INLIERS)))},
        "random_pairs": {"score": pct(rnd_score), "pass": int((rnd_score >= MIN_INLIERS).sum()),
                         "top10": sorted(rnd_score.tolist())[-10:],
                         "ncc_of_first_300": pct(np.round(list(pool.map(lambda p: overlap_ncc(F, *p), rnd[:300])), 3))},
        "retrieved_pairs": {name: {f"{lo}-{lo + 4}": sum(lo <= s < lo + 5 for (n, _, _), s in zip(retrieved, scores, strict=True)
                                                         if n == name) for lo in range(0, 60, 5)}
                            | {">=60": sum(s >= 60 for (n, _, _), s in zip(retrieved, scores, strict=True) if n == name)}
                            for name, _, _ in runs},
    }

    # candidates: one row per pair key (best file pair), minus what dHash already listed
    best_by_key: dict[tuple, tuple[int, str, str, str]] = {}
    near: list[dict] = []  # NEAR <= score < MIN_INLIERS, above every random pair: listed, not put on sheets
    for (name, q, o), s in zip(retrieved, scores, strict=True):
        k = (name, *key(name.removeprefix("kk_"), q, o)[1:])
        if s >= NEAR and (k not in best_by_key or s > best_by_key[k][0]):
            best_by_key[k] = (s, name, q, o)
    for k, (s, name, q, o) in list(best_by_key.items()):
        if s < MIN_INLIERS:
            del best_by_key[k]
            if k not in dh_key:
                near.append({"query": q, "other": o, "set": name, "score": s, "ncc": round(overlap_ncc(F, q, o), 3)})
    st3_key = {key(c["set"], c["kaggle"], c["other"]): c["id"] for c in json.loads(STEP3_CAND.read_text())["external"]}
    piid_out, kk_out = [], []
    for name in ("petchs", "fr7kn", "kaggle", "internal"):
        rows = sorted((v for v in best_by_key.values() if v[1] == name and key(name, v[2], v[3]) not in dh_key),
                      key=lambda v: (-v[0], v[2], v[3]))
        piid_out += [{"id": f"{TAG[name]}{n:03d}", "piid": q, "other": o, "set": name, "score": s,
                      "ncc": round(overlap_ncc(F, q, o), 3)} for n, (s, _, q, o) in enumerate(rows)]
    rows = sorted((v for v in best_by_key.values() if v[1].startswith("kk_")), key=lambda v: (-v[0], v[2], v[3]))
    for n, (s, name, q, o) in enumerate(rows):
        row = {"id": f"kk{n:03d}", "kaggle": q, "other": o, "set": name.removeprefix("kk_"), "score": s,
               "ncc": round(overlap_ncc(F, q, o), 3)}
        if (k := key(row["set"], q, o)) in st3_key:
            row["step3"] = st3_key[k]
        kk_out.append(row)

    SHEETS.mkdir(parents=True, exist_ok=True)
    for old in SHEETS.glob("crop_*.png"):
        old.unlink()
    sheets: dict[str, str] = {}
    groups = [[c for c in piid_out if c["set"] == s] for s in ("petchs", "fr7kn", "kaggle", "internal")] + [kk_out]
    for group in groups:
        pairs_ = [(c["id"], c.get("piid") or c["kaggle"], c["other"], c["score"]) for c in group]
        for n in range(0, len(pairs_), PER_SHEET):
            out = SHEETS / f"crop_{len(sheets):02d}.png"
            sheet(pairs_[n:n + PER_SHEET], out)
            sheets[out.name] = f"{pairs_[n][0]}-{pairs_[min(n + PER_SHEET, len(pairs_)) - 1][0]}"

    counts = {**{s: sum(c["set"] == s for c in piid_out) for s in ("petchs", "fr7kn", "kaggle", "internal")},
              "piid_images_in_new_pairs": len({c["piid"] for c in piid_out} | {c["other"] for c in piid_out if c["set"] == "internal"}),
              "kk_petchs": sum(c["set"] == "petchs" for c in kk_out), "kk_fr7kn": sum(c["set"] == "fr7kn" for c in kk_out),
              "kk_already_seen_by_step3": sum("step3" in c for c in kk_out),
              "kaggle_test_images_in_kk": len({c["kaggle"] for c in kk_out})}
    # sanity check on the dHash pairs: geometric score of each listed pair, and whether this search found it too
    direct = list(pool.map(lambda c: score(F, c["piid"], c["other"]), dh_rows))
    dhash = {s: {"rows": sum(c["set"] == s for c in dh_rows),
                 "score_pass": sum(c["set"] == s and d >= MIN_INLIERS for c, d in zip(dh_rows, direct, strict=True)),
                 "score_pass_and_found": sum(c["set"] == s and d >= MIN_INLIERS and key(s, c["piid"], c["other"]) in best_by_key
                                             for c, d in zip(dh_rows, direct, strict=True))} for s in TAG}
    dhash["score_pass_not_found"] = [c["id"] for c, d in zip(dh_rows, direct, strict=True)
                                     if d >= MIN_INLIERS and key(c["set"], c["piid"], c["other"]) not in best_by_key]
    dhash["confirmed"] = {c["id"]: {"score": d, "found": key(c["set"], c["piid"], c["other"]) in best_by_key}
                          for c, d in zip(dh_rows, direct, strict=True) if c["id"] in CONFIRMED}
    counts["near_threshold"] = dict(Counter(r["set"] for r in near))
    counts["ncc_below_0.5"] = {s: sum(c["ncc"] < 0.5 for c in piid_out if c["set"] == s) for s in TAG} | {
        "kk": sum(c["ncc"] < 0.5 for c in kk_out)}
    counts["near_threshold_ncc_at_least_0.9"] = dict(Counter(r["set"] for r in near if r["ncc"] >= 0.9))
    runtime = {"features_s": round(t_feat), "retrieval_s": round(t_ret), "verification_s": round(t_ver),
               "total_s": round(time.time() - t0)}
    method = {"sift": {"side": SIFT_SIDE, "n": SIFT_N, "contrast": SIFT_CONTRAST},
              "retrieval": {"vote_descriptors": VOTE_N, "flann_checks": CHECKS, "top_k": TOP_K},
              "verification": {"ratio": RATIO, "ransac": "affine", "ransac_px": RANSAC_PX, "min_inliers": MIN_INLIERS}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"method": method, "calibration": calib, "dhash_check": dhash, "counts": counts,
                               "runtime": runtime, "sheets": sheets, "piid": piid_out, "kaggle_leak": kk_out,
                               "near_threshold": sorted(near, key=lambda r: (r["set"], -r["score"]))}, indent=1))
    print(json.dumps({"calibration": calib, "dhash_check": dhash, "counts": counts, "runtime": runtime, "sheets": sheets},
                     indent=1))


if __name__ == "__main__":
    main()
