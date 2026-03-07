from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt


# ==========================================================
# CONFIG (EDIT THIS ONLY)
# ==========================================================
CONFIG = dict(
    input_globs=[
        "Data/**/segmented_headset_data.csv",
    ],
    out_dir="Outputs",

    # Models to run: any subset of ["rf", "ocsvm", "cnn", "transformer"]
    models=[
        "rf",
        "ocsvm",
        "cnn",
        "transformer",
    ],

    hand_side="right",
    exclude_flags={"idle", "pretask", "posttask", "", "nan", "None", "none"},

    use_full_segment=True,   # if False, falls back to fixed-window cropping (window/offset/crops)
    window=60,               # only used when use_full_segment=False
    offset=0,
    crops_per_segment=1,

    max_segments_per_session=None,

    # Denoise switch
    denoise_enabled=False,
    denoise_window=5,
    denoise_center=True,

    # === Sequence length for CNN/Transformer after resampling full segment ===
    seq_len=128,             # frames per sample after resampling for torch models

    # ===== SAME-TYPE 60/20/20 SPLIT =====
    split_train=0.60,
    split_val=0.20,
    split_test=0.20,

    # Recommended grouping (avoid session leakage):
    split_group_key="user_session",   # "user_trial_obj" | "user_trial" | "user_session"

    holdout_mode="same_type_602020",
    include_types=["ball", "bottle", "soda", "projectile"],

    report_fars=[0.02, 0.05, 0.10],

    # Takeover evaluation (no voting by default: agg_k=1)
    do_takeover=True,
    takeover_points=[3, 5],
    agg_k=1,            # 1 = independent decision (no smoothing); set 3 if you want smoothing
    takeover_reps=5,
    takeover_horizon=30,  # cap for plots

    seed=47,
    max_users=None,

    # RF
    rf_trees=300,

    # OCSVM
    ocsvm_nu=0.15,
    ocsvm_gamma="scale",

    # Torch
    nn_epochs=35,
    nn_batch=64,
    nn_lr=1e-3,
    nn_patience=3,
    nn_weight_decay=0.0,

    # CNN
    cnn_filters=[64, 128],
    cnn_kernel=5,
    cnn_dropout=0.1,

    # Transformer
    tx_d_model=64,
    tx_heads=4,
    tx_ff=256,
    tx_layers=2,
    tx_dropout=0.1,

    device="cuda" if torch.cuda.is_available() else "cpu",
)


# ==========================================================
# Helpers
# ==========================================================
def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _infer_user_trial_from_path(path: str) -> Tuple[str, str]:
    parts = re.split(r"[\\/]+", os.path.normpath(path))
    try:
        i = next(i for i, p in enumerate(parts) if p.lower() == "data")
        if i + 2 < len(parts):
            user = parts[i + 1]
            trial = parts[i + 2]
            if re.fullmatch(r"\d+", user) and re.fullmatch(r"\d+", trial):
                return user, trial
    except StopIteration:
        pass

    nums = [p for p in parts if re.fullmatch(r"\d+", p)]
    if len(nums) >= 2:
        return nums[-2], nums[-1]
    if len(nums) == 1:
        return nums[-1], "unknown"
    return "unknown", "unknown"


def _clean_flag_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def denoise_window(w: np.ndarray, win: int = 5, center: bool = True) -> np.ndarray:
    if win is None or int(win) <= 1:
        return w
    dfw = pd.DataFrame(w)
    w2 = dfw.rolling(window=int(win), min_periods=1, center=bool(center)).mean().values
    return w2.astype(np.float32)


def hand_only_columns(df: pd.DataFrame, hand_side: str = "both") -> List[str]:
    drop_prefixes = ("imu_", "hmd_", "head_", "gaze_", "controller", "ctrl_")
    keep = []
    for c in df.columns:
        cl = c.lower()
        if cl == "flag":
            continue
        if any(cl.startswith(p) for p in drop_prefixes):
            continue

        def is_handish(name: str) -> bool:
            return (
                name.startswith("rh_") or name.startswith("lh_") or
                any(k in name for k in ("hand", "finger", "wrist", "palm", "thumb", "index", "middle", "ring", "little", "knuckle", "curl"))
            )

        if hand_side.lower() == "right":
            if cl.startswith("rh_") or ("right" in cl and is_handish(cl)):
                keep.append(c)
        elif hand_side.lower() == "left":
            if cl.startswith("lh_") or ("left" in cl and is_handish(cl)):
                keep.append(c)
        else:
            if is_handish(cl):
                keep.append(c)

    return [c for c in df.columns if c in keep]


# ==========================================================
# Normalization (hand-relative + p5/p95)
# ==========================================================
def _robust_center_scale(series: pd.Series, eps: float = 1e-6) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    x = x.replace([np.inf, -np.inf], np.nan)
    vals = x.values
    if np.all(np.isnan(vals)):
        return pd.Series(np.zeros(len(x), dtype=np.float32), index=series.index)
    med = float(np.nanmedian(vals))
    p5 = float(np.nanpercentile(vals, 5))
    p95 = float(np.nanpercentile(vals, 95))
    scale = max(p95 - p5, eps)
    y = (x - med) / scale
    return y.fillna(0.0).astype(np.float32)


def normalize_coordinates_inplace(df: pd.DataFrame) -> None:
    for side in ["rh", "lh"]:
        hx, hy, hz = f"{side}_pos_x", f"{side}_pos_y", f"{side}_pos_z"
        if not (hx in df.columns and hy in df.columns and hz in df.columns):
            continue

        hand_ref_x = pd.to_numeric(df[hx], errors="coerce")
        hand_ref_y = pd.to_numeric(df[hy], errors="coerce")
        hand_ref_z = pd.to_numeric(df[hz], errors="coerce")

        def _rel(px, py, pz):
            df[px] = pd.to_numeric(df[px], errors="coerce") - hand_ref_x
            df[py] = pd.to_numeric(df[py], errors="coerce") - hand_ref_y
            df[pz] = pd.to_numeric(df[pz], errors="coerce") - hand_ref_z

        fingers = ["thumb", "index", "middle", "ring", "little"]
        for f in fingers:
            px, py, pz = f"{side}_{f}_x", f"{side}_{f}_y", f"{side}_{f}_z"
            if px in df.columns and py in df.columns and pz in df.columns:
                _rel(px, py, pz)
            kx, ky, kz = f"{side}_{f}_knuckle_x", f"{side}_{f}_knuckle_y", f"{side}_{f}_knuckle_z"
            if kx in df.columns and ky in df.columns and kz in df.columns:
                _rel(kx, ky, kz)

    pos_cols = []
    for c in df.columns:
        cl = c.lower()
        if cl.endswith(("_x", "_y", "_z")) and (
            cl.startswith("rh_") or cl.startswith("lh_") or
            any(k in cl for k in ["hand", "finger", "wrist", "palm", "thumb", "index", "middle", "ring", "little", "knuckle", "pos"])
        ):
            pos_cols.append(c)

    for c in pos_cols:
        df[c] = _robust_center_scale(df[c]).clip(-3.0, 3.0)


# ==========================================================
# Segment parsing
# ==========================================================
def segment_object_runs(flags: np.ndarray, exclude_flags: set):
    segs = []
    n = len(flags)
    i = 0
    while i < n:
        f = str(flags[i]).strip()
        if f in exclude_flags:
            i += 1
            continue
        j = i
        while j + 1 < n and str(flags[j + 1]).strip() == f:
            j += 1
        segs.append((i, j, f))
        i = j + 1
    return segs


def crop_windows_in_segment(start: int, end: int, window: int, offset: int, n_crops: int, rng: np.random.Generator):
    seg_len = end - start + 1
    if seg_len < offset + window:
        return []
    low = start + offset
    high = end - window + 1
    if high < low:
        return []
    if n_crops <= 1:
        return [(low, low + window - 1)]
    wins = []
    for _ in range(n_crops):
        s = int(rng.integers(low, high + 1))
        wins.append((s, s + window - 1))
    return wins


# ==========================================================
# Segment -> representations
# ==========================================================
def engineer_features_fullsegment(vals: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(vals.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mu = X.mean(axis=0); sd = X.std(axis=0); mn = X.min(axis=0); mx = X.max(axis=0)
    dX = np.diff(X, axis=0)
    dmu = dX.mean(axis=0) if len(dX) else np.zeros_like(mu)
    dsd = dX.std(axis=0) if len(dX) else np.zeros_like(mu)
    T = np.array([float(X.shape[0])], dtype=np.float32)
    return np.concatenate([mu, sd, mn, mx, dmu, dsd, T], axis=0)


def resample_to_length(X: np.ndarray, L: int) -> np.ndarray:
    X = np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    T, F = X.shape
    if L <= 0:
        raise ValueError("seq_len must be positive.")
    if T == 0:
        return np.zeros((L, F), dtype=np.float32)
    if T == 1:
        return np.repeat(X, repeats=L, axis=0).astype(np.float32)
    if T == L:
        return X.astype(np.float32)

    xp = np.linspace(0.0, 1.0, num=T, dtype=np.float32)
    xq = np.linspace(0.0, 1.0, num=L, dtype=np.float32)
    Y = np.zeros((L, F), dtype=np.float32)
    for f in range(F):
        Y[:, f] = np.interp(xq, xp, X[:, f]).astype(np.float32)
    return Y


@dataclass
class Sample:
    user: str
    trial: str
    session_path: str
    obj: str
    x_feat: np.ndarray
    x_seq: np.ndarray
    seg_len: int


def obj_type_and_variant(obj: str):
    s = str(obj).strip()
    m = re.match(r"(?i)^([a-z]+)[-_]?(\d+)?$", s)
    if m:
        return m.group(1).lower(), (int(m.group(2)) if m.group(2) else None)
    return s.lower(), None


def objects_by_type(objects: List[str]) -> Dict[str, List[str]]:
    groups = defaultdict(list)
    for o in objects:
        t, _ = obj_type_and_variant(o)
        groups[t].append(o)
    for t in groups:
        groups[t] = sorted(groups[t])
    return dict(groups)


def load_samples(paths: List[str]) -> List[Sample]:
    rng = np.random.default_rng(CONFIG["seed"])
    all_samples: List[Sample] = []

    for p in paths:
        df = pd.read_csv(p)
        if "flag" not in df.columns:
            raise ValueError(f"CSV missing 'flag' column: {p}")

        df["flag"] = _clean_flag_series(df["flag"])
        normalize_coordinates_inplace(df)

        cols = hand_only_columns(df, hand_side=CONFIG["hand_side"])
        if not cols:
            raise ValueError(f"No hand-only columns found in {p}.")

        user, trial = _infer_user_trial_from_path(p)
        segs = segment_object_runs(df["flag"].values, CONFIG["exclude_flags"])
        if CONFIG["max_segments_per_session"] is not None:
            segs = segs[:CONFIG["max_segments_per_session"]]

        for (s, e, obj) in segs:
            seg_vals = df.iloc[s:e+1][cols].to_numpy(dtype=np.float32)
            seg_vals = np.nan_to_num(seg_vals, nan=0.0, posinf=0.0, neginf=0.0)

            if CONFIG.get("denoise_enabled", False):
                seg_vals = denoise_window(seg_vals, win=int(CONFIG.get("denoise_window", 3)),
                                          center=bool(CONFIG.get("denoise_center", True)))

            if bool(CONFIG.get("use_full_segment", True)):
                x_feat = engineer_features_fullsegment(seg_vals)
                x_seq = resample_to_length(seg_vals, int(CONFIG["seq_len"]))
                all_samples.append(Sample(user=user, trial=trial, session_path=p, obj=obj,
                                          x_feat=x_feat, x_seq=x_seq, seg_len=int(seg_vals.shape[0])))
            else:
                wins = crop_windows_in_segment(s, e, int(CONFIG["window"]), int(CONFIG["offset"]),
                                               int(CONFIG["crops_per_segment"]), rng)
                for (ws, we) in wins:
                    w = df.iloc[ws:we+1][cols].to_numpy(dtype=np.float32)
                    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
                    if CONFIG.get("denoise_enabled", False):
                        w = denoise_window(w, win=int(CONFIG.get("denoise_window", 3)),
                                           center=bool(CONFIG.get("denoise_center", True)))
                    x_feat = engineer_features_fullsegment(w)
                    x_seq = resample_to_length(w, int(CONFIG["seq_len"]))
                    all_samples.append(Sample(user=user, trial=trial, session_path=p, obj=obj,
                                              x_feat=x_feat, x_seq=x_seq, seg_len=int(w.shape[0])))
    return all_samples


def split_train_val_test_grouped_class_aware(rows: List[Sample], train_ratio: float, val_ratio: float, test_ratio: float,
                                            seed: int, group_key: str, target_user: str):
    if not rows:
        return [], [], []
    rng = np.random.RandomState(int(seed))

    groups: Dict[tuple, List[Sample]] = {}
    for r in rows:
        key = ((r.user, r.trial) if group_key == "user_trial" else
               (r.user, r.session_path) if group_key == "user_session" else
               (r.user, r.trial, r.obj))
        groups.setdefault(key, []).append(r)

    keys = list(groups.keys())
    genu_keys = [k for k in keys if groups[k][0].user == target_user]
    imp_keys = [k for k in keys if groups[k][0].user != target_user]
    rng.shuffle(genu_keys); rng.shuffle(imp_keys)

    def _split_keys(klist):
        n = len(klist)
        if n == 0:
            return [], [], []
        if n == 1:
            return klist, [], []
        if n == 2:
            return [klist[0]], [], [klist[1]]
        n_tr = int(round(train_ratio * n))
        n_va = int(round(val_ratio * n))
        n_tr = max(1, min(n_tr, n-2))
        n_va = max(1, min(n_va, n-n_tr-1))
        tr = klist[:n_tr]
        va = klist[n_tr:n_tr+n_va]
        te = klist[n_tr+n_va:]
        return tr, va, te

    g_tr, g_va, g_te = _split_keys(genu_keys)
    i_tr, i_va, i_te = _split_keys(imp_keys)

    tr_keys = g_tr + i_tr
    va_keys = g_va + i_va
    te_keys = g_te + i_te

    tr = [r for k in tr_keys for r in groups[k]]
    va = [r for k in va_keys for r in groups[k]]
    te = [r for k in te_keys for r in groups[k]]
    return tr, va, te


def build_user_dataset(samples: List[Sample], target_user: str, pool_objs: set):
    pool = [sm for sm in samples if sm.obj in pool_objs]
    if not pool:
        return {}

    train_rows = val_rows = test_rows = []
    best = None

    for attempt in range(50):
        tr, va, te = split_train_val_test_grouped_class_aware(
            pool,
            train_ratio=float(CONFIG["split_train"]),
            val_ratio=float(CONFIG["split_val"]),
            test_ratio=float(CONFIG["split_test"]),
            seed=int(CONFIG["seed"]) + 997 * attempt,
            group_key=str(CONFIG["split_group_key"]),
            target_user=target_user,
        )
        if not tr or not va or not te:
            continue

        ytr_tmp = np.array([1 if r.user == target_user else 0 for r in tr], dtype=int)
        if ytr_tmp.sum() < 5:
            continue

        yva_tmp = np.array([1 if r.user == target_user else 0 for r in va], dtype=int)
        yte_tmp = np.array([1 if r.user == target_user else 0 for r in te], dtype=int)
        if (np.unique(yva_tmp).size == 2) and (np.unique(yte_tmp).size == 2):
            train_rows, val_rows, test_rows = tr, va, te
            best = "good"
            break
        if best is None:
            train_rows, val_rows, test_rows = tr, va, te
            best = "fallback"

    if best is None:
        return {}

    Xtr_f = np.stack([r.x_feat for r in train_rows], axis=0)
    Xva_f = np.stack([r.x_feat for r in val_rows], axis=0)
    Xte_f = np.stack([r.x_feat for r in test_rows], axis=0)

    Xtr_s = np.stack([r.x_seq for r in train_rows], axis=0)
    Xva_s = np.stack([r.x_seq for r in val_rows], axis=0)
    Xte_s = np.stack([r.x_seq for r in test_rows], axis=0)

    ytr = np.array([1 if r.user == target_user else 0 for r in train_rows], dtype=np.int32)
    yva = np.array([1 if r.user == target_user else 0 for r in val_rows], dtype=np.int32)
    yte = np.array([1 if r.user == target_user else 0 for r in test_rows], dtype=np.int32)

    meta_va = pd.DataFrame([{"user": r.user, "trial": r.trial, "obj": r.obj, "path": r.session_path} for r in val_rows])
    meta_te = pd.DataFrame([{"user": r.user, "trial": r.trial, "obj": r.obj, "path": r.session_path} for r in test_rows])
    meta_tr = pd.DataFrame([{"user": r.user, "trial": r.trial, "obj": r.obj, "path": r.session_path} for r in train_rows])

    return dict(
        Xtr_f=Xtr_f, Xva_f=Xva_f, Xte_f=Xte_f,
        Xtr_s=Xtr_s, Xva_s=Xva_s, Xte_s=Xte_s,
        ytr=ytr, yva=yva, yte=yte,
        meta_va=meta_va, meta_te=meta_te, meta_tr=meta_tr,
    )


# ==========================================================
# Metrics
# ==========================================================
def threshold_for_far(y_true: np.ndarray, y_score: np.ndarray, target_far: float) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if np.unique(y_true).size < 2:
        return float(-np.inf) if y_true.mean() > 0.5 else float(np.inf)
    fpr, tpr, thr = roc_curve(y_true, y_score)
    ok = np.where(fpr <= float(target_far))[0]
    i = int(np.argmin(fpr)) if ok.size == 0 else int(ok[np.argmax(tpr[ok])])
    return float(thr[i])


def compute_eer(y_true: np.ndarray, y_score: np.ndarray):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if np.unique(y_true).size < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[i] + fnr[i]) / 2.0)
    return eer, float(thr[i])


def confusion_counts(y_true: np.ndarray, y_score: np.ndarray, thr: float):
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_score) >= float(thr)).astype(int)
    TP = int(((pred == 1) & (y_true == 1)).sum())
    FN = int(((pred == 0) & (y_true == 1)).sum())
    FP = int(((pred == 1) & (y_true == 0)).sum())
    TN = int(((pred == 0) & (y_true == 0)).sum())
    return TP, FN, FP, TN


# ==========================================================
# Takeover
# ==========================================================
def takeover_time_to_detect(meta_te: pd.DataFrame, y_score: np.ndarray, target_user: str, thr: float,
                            takeover_at: int, agg_k: int, rng: np.random.Generator):
    df = meta_te.copy()
    df["score"] = y_score
    genu = df[df["user"] == target_user].copy()
    imps = df[df["user"] != target_user].copy()
    if len(genu) <= takeover_at or len(imps) < 1:
        return None

    imp_user = str(rng.choice(imps["user"].unique()))
    imp_sel = imps[imps["user"] == imp_user].copy()

    genu = genu.sort_values(["path", "trial"]).reset_index(drop=True)
    imp_sel = imp_sel.sort_values(["path", "trial"]).reset_index(drop=True)

    post_len = max(1, len(genu) - takeover_at)
    post_len = min(post_len, len(imp_sel))
    stream = pd.concat([genu.iloc[:takeover_at], imp_sel.iloc[:post_len]], ignore_index=True)

    det = np.inf
    for i in range(len(stream)):
        w = stream["score"].iloc[max(0, i - agg_k + 1): i + 1].values
        s = float(np.mean(w))
        if i >= takeover_at and (s < thr):
            det = float(i - takeover_at + 1)
            break

    return {
        "imp_user": imp_user,
        "stream_len": int(len(stream)),
        "takeover_at": int(takeover_at),
        "agg_k": int(agg_k),
        "ttd": float(det),
    }


# ==========================================================
# Models
# ==========================================================
def fit_rf(Xtr_f: np.ndarray, ytr: np.ndarray, Xva_f: np.ndarray, Xte_f: np.ndarray):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr_f)
    Xva = scaler.transform(Xva_f)
    Xte = scaler.transform(Xte_f)
    clf = RandomForestClassifier(
        n_estimators=int(CONFIG["rf_trees"]),
        random_state=int(CONFIG["seed"]),
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    clf.fit(Xtr, ytr)
    s_tr = clf.predict_proba(Xtr)[:, 1]
    s_va = clf.predict_proba(Xva)[:, 1]
    s_te = clf.predict_proba(Xte)[:, 1]
    return s_va, s_te, s_tr


def fit_ocsvm(Xtr_f: np.ndarray, ytr: np.ndarray, Xva_f: np.ndarray, Xte_f: np.ndarray):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr_f)
    Xva = scaler.transform(Xva_f)
    Xte = scaler.transform(Xte_f)
    Xg = Xtr[ytr == 1]
    if len(Xg) < 10:
        raise ValueError("Not enough genuine samples for OCSVM.")
    oc = OneClassSVM(kernel="rbf", nu=float(CONFIG["ocsvm_nu"]), gamma=CONFIG["ocsvm_gamma"])
    oc.fit(Xg)
    s_va = oc.decision_function(Xva).ravel()
    s_te = oc.decision_function(Xte).ravel()
    s_tr = oc.decision_function(Xtr).ravel()
    lo, hi = np.percentile(s_tr, 1), np.percentile(s_tr, 99)
    hi = hi if hi > lo else (lo + 1e-6)
    yscore_va = np.clip((s_va - lo) / (hi - lo), 0.0, 1.0)
    yscore_te = np.clip((s_te - lo) / (hi - lo), 0.0, 1.0)
    yscore_tr = np.clip((s_tr - lo) / (hi - lo), 0.0, 1.0)
    return yscore_va, yscore_te, yscore_tr


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
    def __len__(self):
        return int(self.X.shape[0])
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def standardize_sequences_trainvaltest(Xtr_s: np.ndarray, Xva_s: np.ndarray, Xte_s: np.ndarray):
    N, T, Fdim = Xtr_s.shape
    scaler = StandardScaler()
    scaler.fit(Xtr_s.reshape(-1, Fdim))
    Xtr = scaler.transform(Xtr_s.reshape(-1, Fdim)).reshape(N, T, Fdim).astype(np.float32)
    Xva = scaler.transform(Xva_s.reshape(-1, Fdim)).reshape(Xva_s.shape[0], Xva_s.shape[1], Fdim).astype(np.float32)
    Xte = scaler.transform(Xte_s.reshape(-1, Fdim)).reshape(Xte_s.shape[0], Xte_s.shape[1], Fdim).astype(np.float32)
    return Xtr, Xva, Xte


class CNN1D(nn.Module):
    def __init__(self, T: int, Fdim: int, filters: List[int], kernel: int, dropout: float):
        super().__init__()
        layers = []
        in_ch = Fdim
        for out_ch in filters:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=int(kernel), padding=int(kernel)//2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(float(dropout)),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_ch, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.head(x)
        return x.squeeze(-1)


class TransformerClassifier(nn.Module):
    def __init__(self, T: int, Fdim: int, d_model: int, nhead: int, dim_ff: int, num_layers: int, dropout: float):
        super().__init__()
        self.T = int(T)
        self.proj = nn.Linear(Fdim, int(d_model))
        self.pos_emb = nn.Embedding(self.T, int(d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=int(dim_ff),
            dropout=float(dropout),
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.mlp = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), 128),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        B, T, _ = x.shape
        if T != self.T:
            raise ValueError(f"Transformer expected T={self.T}, got T={T}")
        h = self.proj(x)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = h + self.pos_emb(pos)
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.mlp(h).squeeze(-1)


@torch.no_grad()
def torch_predict_proba(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    outs = []
    for xb, _ in loader:
        xb = xb.to(device)
        logits = model(xb)
        outs.append(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1))
    return np.concatenate(outs, axis=0) if outs else np.zeros((0,), dtype=np.float32)


def fit_torch_binary(model: nn.Module, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, Xte: np.ndarray):
    device = str(CONFIG["device"])
    model = model.to(device)
    tr_loader = DataLoader(SeqDataset(Xtr, ytr), batch_size=int(CONFIG["nn_batch"]), shuffle=True)
    tr_eval_loader = DataLoader(SeqDataset(Xtr, ytr), batch_size=int(CONFIG["nn_batch"]), shuffle=False)
    va_loader = DataLoader(SeqDataset(Xva, yva), batch_size=int(CONFIG["nn_batch"]), shuffle=False)
    te_loader = DataLoader(SeqDataset(Xte, np.zeros(len(Xte), dtype=np.float32)), batch_size=int(CONFIG["nn_batch"]), shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=float(CONFIG["nn_lr"]), weight_decay=float(CONFIG["nn_weight_decay"]))
    criterion = nn.BCEWithLogitsLoss()
    best_val = float("inf")
    best_state = None
    bad = 0
    for _ in range(int(CONFIG["nn_epochs"])):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
        model.eval()
        vloss_sum = 0.0; vcount = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device); yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                vloss_sum += float(loss.item()) * int(xb.size(0))
                vcount += int(xb.size(0))
        vloss = vloss_sum / max(vcount, 1)
        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(CONFIG["nn_patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    s_tr = torch_predict_proba(model, tr_eval_loader, device=device)
    s_va = torch_predict_proba(model, va_loader, device=device)
    s_te = torch_predict_proba(model, te_loader, device=device)
    return s_va, s_te, s_tr

# ==========================================================
# Main
# ==========================================================
def main():
    set_seed(int(CONFIG["seed"]))
    rng = np.random.default_rng(int(CONFIG["seed"]))

    paths: List[str] = []
    for pat in CONFIG["input_globs"]:
        paths.extend(glob.glob(pat, recursive=True))
    paths = sorted(list(dict.fromkeys(paths)))
    if not paths:
        raise SystemExit("No input files matched. Edit CONFIG['input_globs'].")

    print(f"[INFO] Found {len(paths)} session CSVs.")
    print(f"[CFG] use_full_segment={CONFIG['use_full_segment']} | denoise={CONFIG['denoise_enabled']} | seq_len={CONFIG['seq_len']} | agg_k={CONFIG['agg_k']}")

    samples = load_samples(paths)
    print(f"[INFO] Built {len(samples)} interaction samples from object segments.")

    objects = sorted(list({s.obj for s in samples}))
    users = sorted(list({s.user for s in samples if s.user != "unknown"}))
    if CONFIG["max_users"] is not None:
        users = users[:int(CONFIG["max_users"])]

    by_type = objects_by_type(objects)
    wanted = [str(t).lower() for t in (CONFIG.get("include_types") or [])]
    if not wanted:
        wanted = sorted(by_type.keys())

    holdouts = []
    for t in wanted:
        if t in by_type and len(by_type[t]) >= 2:
            holdouts.append((t, set(by_type[t])))

    if not holdouts:
        raise SystemExit(f"No types matched include_types={wanted}. Available types: {sorted(by_type.keys())}")

    out_dir = str(CONFIG["out_dir"])
    os.makedirs(out_dir, exist_ok=True)

    for model_name in CONFIG["models"]:
        print(f"\n[MODEL] {model_name}")
        summary_rows = []
        takeover_rows = []
        pooled_te_rows = []
        pooled_va_rows = []
        pooled_tr_rows = []

        for (type_name, pool_objs) in holdouts:
            for u in users:
                pack = build_user_dataset(samples, target_user=u, pool_objs=pool_objs)
                if not pack:
                    continue

                Xtr_f, Xva_f, Xte_f = pack["Xtr_f"], pack["Xva_f"], pack["Xte_f"]
                Xtr_s, Xva_s, Xte_s = pack["Xtr_s"], pack["Xva_s"], pack["Xte_s"]
                ytr, yva, yte = pack["ytr"], pack["yva"], pack["yte"]
                meta_va = pack["meta_va"]
                meta_te = pack["meta_te"]
                meta_tr = pack["meta_tr"]
                

                try:
                    if model_name == "rf":
                        s_va, s_te, s_tr = fit_rf(Xtr_f, ytr, Xva_f, Xte_f)
                    elif model_name == "ocsvm":
                        s_va, s_te, s_tr = fit_ocsvm(Xtr_f, ytr, Xva_f, Xte_f)
                    elif model_name == "cnn":
                        Xtr, Xva, Xte = standardize_sequences_trainvaltest(Xtr_s, Xva_s, Xte_s)
                        _, T, Fdim = Xtr.shape
                        model = CNN1D(T=T, Fdim=Fdim, filters=list(CONFIG["cnn_filters"]),
                                      kernel=int(CONFIG["cnn_kernel"]), dropout=float(CONFIG["cnn_dropout"]))
                        s_va, s_te, s_tr = fit_torch_binary(model, Xtr, ytr.astype(np.float32), Xva, yva.astype(np.float32), Xte)
                    elif model_name == "transformer":
                        Xtr, Xva, Xte = standardize_sequences_trainvaltest(Xtr_s, Xva_s, Xte_s)
                        _, T, Fdim = Xtr.shape
                        model = TransformerClassifier(T=T, Fdim=Fdim,
                                                      d_model=int(CONFIG["tx_d_model"]),
                                                      nhead=int(CONFIG["tx_heads"]),
                                                      dim_ff=int(CONFIG["tx_ff"]),
                                                      num_layers=int(CONFIG["tx_layers"]),
                                                      dropout=float(CONFIG["tx_dropout"]))
                        s_va, s_te, s_tr = fit_torch_binary(model, Xtr, ytr.astype(np.float32), Xva, yva.astype(np.float32), Xte)
                    else:
                        raise ValueError(f"Unknown model: {model_name}")
                except Exception as e:
                    print(f"[SKIP] model={model_name} type={type_name} user={u} reason={type(e).__name__}: {e}")
                    continue

                auc = float(roc_auc_score(yte, s_te)) if len(np.unique(yte)) > 1 else float("nan")
                eer, eer_thr = compute_eer(yte, s_te)

                thr_oper = threshold_for_far(yva, s_va, 0.02)
                TP, FN, FP, TN = confusion_counts(yte, s_te, thr_oper)

                tar_table = {}
                for f in CONFIG["report_fars"]:
                    thr_f = threshold_for_far(yva, s_va, float(f))
                    TPv, FNv, FPv, TNv = confusion_counts(yva, s_va, thr_f)                    
                    TPf, FNf, FPf, TNf = confusion_counts(yte, s_te, thr_f)
                    tar_table[f"tar@far{float(f):.3f}"] = float(TPf / max(TPf + FNf, 1))

                summary_rows.append({
                    "model": model_name,
                    "user": u,
                    "test_objs": type_name,
                    "n_val": int(len(yva)),
                    "n_test": int(len(yte)),
                    "AUC": auc,
                    "EER": float(eer),
                    "EER_thr": float(eer_thr),
                    "thr@far0.02": float(thr_oper),
                    "TP": int(TP), "FN": int(FN), "FP": int(FP), "TN": int(TN),
                    **tar_table,
                })

                for i in range(len(yva)):
                    pooled_va_rows.append({
                        "model": model_name,
                        "user_target": u,
                        "true_user": str(meta_va.iloc[i]["user"]),
                        "test_obj": str(meta_va.iloc[i]["obj"]),
                        "type": type_name,
                        "path": str(meta_va.iloc[i]["path"]),
                        "y_true": int(yva[i]),
                        "y_score": float(s_va[i]),
                    })

                for i in range(len(yte)):
                    pooled_te_rows.append({
                        "model": model_name,
                        "user_target": u,
                        "true_user": str(meta_te.iloc[i]["user"]),
                        "test_obj": str(meta_te.iloc[i]["obj"]),
                        "type": type_name,
                        "path": str(meta_te.iloc[i]["path"]),
                        "y_true": int(yte[i]),
                        "y_score": float(s_te[i]),
                    })
                for i in range(len(ytr)):
                    pooled_tr_rows.append({
                        "model": model_name,
                        "user_target": u,
                        "true_user": str(meta_tr.iloc[i]["user"]),
                        "test_obj": str(meta_tr.iloc[i]["obj"]),
                        "type": type_name,
                        "path": str(meta_tr.iloc[i]["path"]),
                        "y_true": int(ytr[i]),
                        "y_score": float(s_tr[i]),
                    })

                if CONFIG["do_takeover"]:
                    thr = threshold_for_far(yva, s_va, 0.02)
                    for t_at in CONFIG["takeover_points"]:
                        for rep in range(int(CONFIG["takeover_reps"])):
                            tk = takeover_time_to_detect(meta_te, s_te, u, thr, int(t_at), int(CONFIG["agg_k"]), rng)
                            if tk is None:
                                continue
                            takeover_rows.append({
                                "model": model_name,
                                "user": u,
                                "test_objs": type_name,
                                "rep": int(rep),
                                **tk,
                            })

        summary_df = pd.DataFrame(summary_rows)
        pooled_te_df = pd.DataFrame(pooled_te_rows)
        pooled_va_df = pd.DataFrame(pooled_va_rows)
        pooled_tr_df = pd.DataFrame(pooled_tr_rows)
        takeover_df = pd.DataFrame(takeover_rows)

        # Save CSVs
        summary_path = os.path.join(out_dir, f"summary_metrics_{model_name}.csv")
        pooled_te_path = os.path.join(out_dir, f"pooled_test_scores_{model_name}.csv")
        pooled_va_path = os.path.join(out_dir, f"pooled_val_scores_{model_name}.csv")
        pooled_tr_path = os.path.join(out_dir, f"pooled_train_scores_{model_name}.csv")
        takeover_path = os.path.join(out_dir, f"takeover_metrics_{model_name}.csv")

        summary_df.to_csv(summary_path, index=False)
        pooled_te_df.to_csv(pooled_te_path, index=False)
        pooled_va_df.to_csv(pooled_va_path, index=False)
        pooled_tr_df.to_csv(pooled_tr_path, index=False)
        print(f"[SAVED] {summary_path}")
        print(f"[SAVED] {pooled_va_path}")
        print(f"[SAVED] {pooled_te_path}")
        print(f"[SAVED] {pooled_tr_path}")

        if CONFIG["do_takeover"]:
            takeover_df.to_csv(takeover_path, index=False)
            print(f"[SAVED] {takeover_path}")

        # Console summary
        if not summary_df.empty:
            cols = ["AUC", "EER"] + [f"tar@far{float(f):.3f}" for f in CONFIG["report_fars"]]
            agg = summary_df.groupby(["test_objs"])[cols].mean().reset_index()
            print("\n=== Mean over users (by type/task family) ===")
            print(agg.to_string(index=False))

    print("\n[DONE] All models complete.")


if __name__ == "__main__":
    main()
