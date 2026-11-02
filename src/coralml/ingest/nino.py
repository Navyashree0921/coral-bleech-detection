"""Niño 3.4 monthly anomalies (NOAA PSL standard format) and a leakage-safe weekly join."""
import numpy as np
import pandas as pd
import requests


def parse_psl_standard(text):
    """Lines: 'year v1 ... v12'. Header line has first/last year; footer is text. Missing = -99.x."""
    recs = {}
    for line in text.splitlines():
        p = line.split()
        if len(p) == 13:
            try:
                y = int(p[0]); vals = [float(x) for x in p[1:]]
            except ValueError:
                continue
            if 1800 < y < 2200:
                for m, v in enumerate(vals, start=1):
                    recs[pd.Timestamp(year=y, month=m, day=1)] = np.nan if v <= -90 else v
    return pd.Series(recs, name="nino34").sort_index()


def download(url, timeout=120):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def join_weekly(weeks, monthly, lag_months=1):
    """Each week gets the Niño 3.4 value of the month `lag_months` before that week's own month.
    A month's index is only published after the month ends, so lag_months >= 1 avoids leakage."""
    key = (weeks.dt.to_period("M") - lag_months).dt.to_timestamp()
    return key.map(monthly)
