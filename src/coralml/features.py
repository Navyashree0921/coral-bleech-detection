"""Feature and label construction. Every feature at week t uses data up to t only."""
import numpy as np
import pandas as pd

N_LAGS = 12  # weeks of history kept as lag columns (also the sequence length)


def add_features(raw: pd.DataFrame) -> pd.DataFrame:
    """raw columns: site, date (weekly), sst, mmm (max monthly mean), nino34."""
    df = raw.sort_values(["site", "date"]).reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["week"] = df["date"].dt.isocalendar().week.astype(int).clip(upper=52)
    df["hotspot"] = df["sst"] - df["mmm"]
    g = df.groupby("site", sort=False)

    # Degree heating weeks: sum of hotspots >= 1 C over the last 12 weeks (C-weeks).
    if "dhw" not in df.columns:   # real data supplies NOAA's own DHW; synthetic data does not
        hs_pos = df["hotspot"].where(df["hotspot"] >= 1.0, 0.0)
        df["dhw"] = hs_pos.groupby(df["site"]).transform(lambda s: s.rolling(12, min_periods=12).sum())

    cols = {}
    for k in range(N_LAGS):
        cols[f"hotspot_lag{k}"] = g["hotspot"].shift(k)
        cols[f"nino34_lag{k}"] = g["nino34"].shift(k)
    cols["hotspot_roll4"] = g["hotspot"].transform(lambda s: s.rolling(4).mean())
    cols["hotspot_roll8"] = g["hotspot"].transform(lambda s: s.rolling(8).mean())
    cols["hotspot_trend4"] = df["hotspot"] - g["hotspot"].shift(4)
    cols["nino34_chg8"] = df["nino34"] - g["nino34"].shift(8)
    cols["woy_sin"] = np.sin(2 * np.pi * df["week"] / 52)
    cols["woy_cos"] = np.cos(2 * np.pi * df["week"] / 52)
    df = pd.concat([df, pd.DataFrame(cols)], axis=1)
    return df.dropna(subset=["dhw"] + [f"hotspot_lag{N_LAGS-1}"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame, use_enso: bool = True) -> list:
    base = ["dhw", "week", "woy_sin", "woy_cos", "hotspot_roll4", "hotspot_roll8", "hotspot_trend4"]
    base += [f"hotspot_lag{k}" for k in range(N_LAGS)]
    if use_enso:
        base += ["nino34", "nino34_chg8"] + [f"nino34_lag{k}" for k in range(N_LAGS)]
    cols = [c for c in base if c in df.columns]
    return cols


def add_labels(df: pd.DataFrame, horizon_weeks: int, alert_dhw: float = 4.0,
               onset_only: bool = False) -> pd.DataFrame:
    """label = 1 if DHW reaches alert_dhw at any of weeks t+1 .. t+horizon (never uses week t)."""
    def future_max(s):
        return s[::-1].rolling(horizon_weeks, min_periods=horizon_weeks).max()[::-1].shift(-1)
    out = df.copy()
    out["future_dhw_max"] = out.groupby("site", sort=False)["dhw"].transform(future_max)
    out = out.dropna(subset=["future_dhw_max"]).copy()
    out["label"] = (out["future_dhw_max"] >= alert_dhw).astype(int)
    if onset_only:  # only ask about reefs not already in alert
        out = out[out["dhw"] < alert_dhw]
    return out.reset_index(drop=True)
