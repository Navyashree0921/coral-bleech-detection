"""LSTM / GRU / 1D-CNN on the last T weeks of hotspot (and Nino 3.4 when available)."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .features import N_LAGS


def torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


class SeqClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, kind="lstm", hidden=32, lr=0.003, epochs=12, batch=256, seed=0, T=N_LAGS):
        self.kind, self.hidden, self.lr, self.epochs = kind, hidden, lr, epochs
        self.batch, self.seed, self.T = batch, seed, T

    def _tensor(self, X):
        vars_ = [v for v in ("hotspot", "nino34") if f"{v}_lag0" in X.columns]
        # oldest week first, newest last
        arr = np.stack([np.column_stack([X[f"{v}_lag{k}"].values for k in range(self.T - 1, -1, -1)])
                        for v in vars_], axis=-1).astype("float32")
        return arr

    def fit(self, X, y):
        import torch, torch.nn as nn
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.classes_ = np.array([0, 1])
        A = self._tensor(X)
        self.mu_, self.sd_ = A.mean((0, 1), keepdims=True), A.std((0, 1), keepdims=True) + 1e-6
        A = (A - self.mu_) / self.sd_
        F = A.shape[-1]
        kind, H = self.kind, int(self.hidden)

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                if kind == "cnn1d":
                    s.conv = nn.Sequential(nn.Conv1d(F, H, 3, padding=1), nn.ReLU(), nn.AdaptiveMaxPool1d(1))
                else:
                    s.rnn = (nn.LSTM if kind == "lstm" else nn.GRU)(F, H, batch_first=True)
                s.out = nn.Linear(H, 1)

            def forward(s, x):
                if kind == "cnn1d":
                    h = s.conv(x.transpose(1, 2)).squeeze(-1)
                else:
                    h = s.rnn(x)[0][:, -1]
                return s.out(h).squeeze(-1)

        self.net_ = Net()
        yv = np.asarray(y, dtype="float32")
        pw = torch.tensor((1 - yv.mean()) / max(yv.mean(), 1e-3))
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        Xt, yt = torch.tensor(A), torch.tensor(yv)
        for _ in range(self.epochs):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), self.batch):
                b = perm[i:i + self.batch]
                opt.zero_grad()
                lossf(self.net_(Xt[b]), yt[b]).backward()
                opt.step()
        return self

    def predict_proba(self, X):
        import torch
        A = (self._tensor(X) - self.mu_) / self.sd_
        self.net_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net_(torch.tensor(A))).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
