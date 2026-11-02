"""SYNTHETIC reef data for testing the pipeline only.
ENSO is built to influence SST here, so any 'El Nino helps' result on this data means nothing.
Never report results from synthetic data."""
import numpy as np
import pandas as pd


def make_synthetic(n_sites=6, start="2000-01-02", end="2024-12-29", seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="7D")
    T = len(dates)
    t = np.arange(T)
    nino = np.zeros(T)
    for i in range(1, T):
        nino[i] = 0.985 * nino[i - 1] + rng.normal(0, 0.12)
    nino += 1.0 * np.sin(2 * np.pi * t / (52 * 4.2) + 1.0)
    rows = []
    week = np.minimum(dates.isocalendar().week.to_numpy().astype(int), 52)
    for s in range(n_sites):
        base = 27.0 + rng.uniform(-0.8, 0.8)
        amp = 1.6 + rng.uniform(-0.3, 0.3)
        season = base + amp * np.sin(2 * np.pi * (week - 10) / 52)
        mmm = base + amp * 0.97
        sens = rng.uniform(0.7, 1.1)
        noise = np.zeros(T)
        for i in range(1, T):
            noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 0.18)
        lagged_nino = np.concatenate([np.zeros(8), nino[:-8]])
        sst = season + 0.55 * sens * lagged_nino + noise + 0.02 * (t / 52)
        rows.append(pd.DataFrame({"site": f"site_{s}", "date": dates, "sst": sst,
                                  "mmm": mmm, "nino34": nino, "region": f"region_{s % 3}"}))
    return pd.concat(rows, ignore_index=True)
