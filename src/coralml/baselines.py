import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class PersistenceClassifier(BaseEstimator, ClassifierMixin):
    """Score = current heat stress relative to the alert threshold (stress keeps going)."""
    def __init__(self, alert_dhw=4.0):
        self.alert_dhw = alert_dhw

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.clip(np.asarray(X["dhw"], dtype=float) / self.alert_dhw, 0, 1)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ClimatologyClassifier(BaseEstimator, ClassifierMixin):
    """Probability = share of training years with an alert in that week of the year."""
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        y = np.asarray(y)
        week = np.asarray(X["week"])
        self.mean_ = float(y.mean())
        self.table_ = {int(w): float(y[week == w].mean()) for w in np.unique(week)}
        return self

    def predict_proba(self, X):
        p = np.array([self.table_.get(int(w), self.mean_) for w in np.asarray(X["week"])])
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
