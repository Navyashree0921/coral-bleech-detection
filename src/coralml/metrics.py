"""Metrics suited to rare events."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve


def _ok(y):
    return len(np.unique(y)) == 2


def pr_auc(y, p):
    return average_precision_score(y, p) if _ok(y) else np.nan


def roc_auc(y, p):
    return roc_auc_score(y, p) if _ok(y) else np.nan


def brier(y, p):
    return brier_score_loss(y, np.clip(p, 0, 1))


def ece(y, p, bins=10):
    p = np.clip(p, 0, 1)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return total


def recall_at_precision(y, p, floor=0.5):
    if not _ok(y):
        return np.nan
    prec, rec, _ = precision_recall_curve(y, p)
    ok = prec >= floor
    return float(rec[ok].max()) if ok.any() else 0.0


def all_metrics(y, p, precision_floor=0.5):
    y = np.asarray(y); p = np.asarray(p, dtype=float)
    return dict(pr_auc=pr_auc(y, p), roc_auc=roc_auc(y, p), brier=brier(y, p),
                ece=ece(y, p), recall_at_prec=recall_at_precision(y, p, precision_floor),
                pos_rate=float(y.mean()))


def year_bootstrap_ci(y, p, years, metric=pr_auc, reps=200, seed=0):
    """Resample whole years (events are clustered in time, so row resampling would be too optimistic)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p); years = np.asarray(years)
    uy = np.unique(years)
    vals = []
    for _ in range(reps):
        pick = rng.choice(uy, size=len(uy), replace=True)
        idx = np.concatenate([np.where(years == yy)[0] for yy in pick])
        v = metric(y[idx], p[idx])
        if not np.isnan(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)
