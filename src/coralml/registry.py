"""Build any model in configs/models.yaml, and sample hyperparameters for random search."""
import importlib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .baselines import PersistenceClassifier, ClimatologyClassifier
from .seq import SeqClassifier, torch_available

LIB = {"xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost",
       "lstm": "torch", "gru": "torch", "cnn1d": "torch"}


def is_available(kind):
    return kind not in LIB or importlib.util.find_spec(LIB[kind]) is not None


def sample_params(space, rng):
    out = {}
    for name, spec in (space or {}).items():
        kind = spec[0]
        if kind == "int":
            out[name] = int(rng.integers(spec[1], spec[2] + 1))
        elif kind == "float":
            out[name] = float(rng.uniform(spec[1], spec[2]))
        elif kind == "logfloat":
            out[name] = float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
        elif kind == "choice":
            out[name] = spec[1][int(rng.integers(len(spec[1])))]
    return out


class Subsampled(BaseEstimator, ClassifierMixin):
    """Caps training rows for slow models (SVM, kNN) so benchmarks stay affordable."""
    def __init__(self, est=None, max_rows=12000, seed=0):
        self.est, self.max_rows, self.seed = est, max_rows, seed

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        y = np.asarray(y)
        if len(y) > self.max_rows:
            idx = np.random.default_rng(self.seed).choice(len(y), self.max_rows, replace=False)
            X, y = X.iloc[idx], y[idx]
        self.m_ = clone(self.est).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m_.predict_proba(X)

    def predict(self, X):
        return self.m_.predict(X)


def make_model(kind, params, seed, y_train):
    p = dict(params or {})
    spw = float((1 - np.mean(y_train)) / max(np.mean(y_train), 1e-3))
    if kind == "persistence":
        return PersistenceClassifier()
    if kind == "climatology":
        return ClimatologyClassifier()
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", **p))
    if kind == "knn":
        return Subsampled(make_pipeline(StandardScaler(), KNeighborsClassifier(**p)), seed=seed)
    if kind == "naive_bayes":
        return make_pipeline(StandardScaler(), GaussianNB())
    if kind == "svm":
        return Subsampled(make_pipeline(StandardScaler(), SVC(probability=True, class_weight="balanced",
                                                              random_state=seed, **p)), max_rows=6000, seed=seed)
    if kind == "decision_tree":
        return DecisionTreeClassifier(class_weight="balanced", random_state=seed, **p)
    if kind == "random_forest":
        return RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=seed, **p)
    if kind == "extra_trees":
        return ExtraTreesClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=seed, **p)
    if kind == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed, **p)
    if kind == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(scale_pos_weight=spw, random_state=seed, n_jobs=-1, eval_metric="logloss", **p)
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(scale_pos_weight=spw, random_state=seed, n_jobs=-1, verbose=-1, **p)
    if kind == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(auto_class_weights="Balanced", random_seed=seed, verbose=0,
                                  thread_count=-1, **p)
    if kind == "mlp":
        h = int(p.pop("hidden", 64))
        return make_pipeline(StandardScaler(), MLPClassifier((h, h // 2), max_iter=200, early_stopping=True,
                                                             random_state=seed, **p))
    if kind in ("lstm", "gru", "cnn1d"):
        return SeqClassifier(kind=kind, seed=seed, **p)
    raise ValueError(f"unknown model kind: {kind}")
