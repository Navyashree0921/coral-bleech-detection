"""Tests use a simulated ERDDAP server (no internet needed). They check parsing, land handling,
chunking, weekly sampling, the Niño lag, and that the output feeds straight into the benchmark."""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest
import yaml

from coralml.ingest import crw, nino
from coralml.ingest.build import build_all, check_dhw
from coralml.features import add_features, add_labels, feature_columns

DAYS = pd.date_range("1985-04-01", "2026-10-15", freq="D")
doy = DAYS.dayofyear.values
SST = 28.6 + 1.2 * np.sin(2 * np.pi * (doy - 100) / 365) + 0.9 * np.sin(2 * np.pi * np.arange(len(DAYS)) / (365 * 4.5))
HOT = SST - 29.2
HS_POS = np.where(HOT >= 1.0, HOT, 0.0)
DHW = pd.Series(HS_POS).rolling(84, min_periods=1).sum().values / 7.0
LAND_LAT = 10.575          # one probe pixel is "land"


class FakeResp:
    def __init__(self, text, code=200):
        self.text, self.status_code = text, code


def fake_get(url, timeout=None):
    m = re.search(r"\[\((\d{4}-\d{2}-\d{2})T12:00:00Z\):1:\((\d{4}-\d{2}-\d{2})T12:00:00Z\)\]\[\(([-\d.]+)\)\]\[\(([-\d.]+)\)\]", url)
    t0, t1, lat, lon = m.group(1), m.group(2), float(m.group(3)), float(m.group(4))
    vars_ = re.findall(r"(CRW_[A-Z]+)\[", url)
    sel = (DAYS >= t0) & (DAYS <= t1)
    cols = {"CRW_SST": SST, "CRW_HOTSPOT": HOT, "CRW_DHW": DHW}
    land = abs(lat - LAND_LAT) < 1e-6
    lines = ["time,latitude,longitude," + ",".join(vars_),
             "UTC,degrees_north,degrees_east," + ",".join(["Celsius"] * len(vars_))]
    for i in np.where(sel)[0]:
        vals = ["" if land else f"{cols[v][i]:.4f}" for v in vars_]
        lines.append(f"{DAYS[i].strftime('%Y-%m-%d')}T12:00:00Z,{lat},{lon}," + ",".join(vals))
    return FakeResp("\n".join(lines) + "\n")


@pytest.fixture(autouse=True)
def patch_get(monkeypatch):
    monkeypatch.setattr(crw.requests, "get", fake_get)


CFG = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs/data.yaml"))


def test_url_and_parse():
    url = crw.build_url(CFG["crw"]["erddap_url"], ["CRW_SST", "CRW_DHW"], "2020-01-01", "2020-01-10", 10.575, 72.675)
    assert "CRW_SST[(2020-01-01T12:00:00Z):1:(2020-01-10T12:00:00Z)][(10.575)][(72.675)]" in url
    df = crw.parse_erddap_csv(fake_get(url).text)
    assert len(df) == 10 and list(df.columns) == ["CRW_SST", "CRW_DHW"] and df.index[0] == pd.Timestamp("2020-01-01")


def test_land_pixel_falls_back_to_nearby_water():
    lat, lon = crw.find_water_pixel(CFG["crw"], 10.57, 72.64, log=lambda *_: None)   # snaps to 10.575 = land
    assert abs(lat - LAND_LAT) > 1e-6 and abs(lat - 10.575) <= 0.16


def test_chunked_download_has_no_gaps_or_duplicates():
    cfg = dict(CFG["crw"]); cfg["start"], cfg["end"], cfg["chunk_years"] = "1990-01-01", "2004-12-31", 5
    d = crw.download_site(cfg, 10.675, 72.675, log=lambda *_: None)
    assert d.index.is_unique and len(d) == len(pd.date_range("1990-01-01", "2004-12-31"))


def test_weekly_sampling_ends_on_sunday():
    d = crw.download_site({**CFG["crw"], "start": "2000-01-01", "end": "2001-12-31"}, 10.675, 72.675, log=lambda *_: None)
    w = crw.to_weekly(d)
    assert (w.index.dayofweek == 6).all()


NINO = """ 1950 2026
 2025 -0.5 -0.4 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
 2026 1.0 1.1 1.2 1.3 1.4 1.5 -99.99 -99.99 -99.99 -99.99 -99.99 -99.99
 -99.99
 NOAA footer text here
"""


def test_nino_parse_and_missing():
    s = nino.parse_psl_standard(NINO)
    assert s[pd.Timestamp("2025-01-01")] == -0.5 and np.isnan(s[pd.Timestamp("2026-07-01")])


def test_nino_lag_uses_previous_completed_month():
    s = nino.parse_psl_standard(NINO)
    weeks = pd.Series(pd.to_datetime(["2026-03-01", "2026-03-29", "2026-04-05"]))
    got = nino.join_weekly(weeks, s, lag_months=1).values
    assert list(got) == [1.1, 1.1, 1.2]         # March weeks see February (1.1); April sees March (1.2)
    assert s[pd.Timestamp("2026-03-01")] == 1.2  # the March value itself is never used in March


def test_end_to_end_feeds_benchmark_code():
    cfg = dict(CFG)
    cfg["crw"] = {**CFG["crw"], "start": "1995-01-01", "end": "2012-12-31"}
    cfg["regions"] = {"lakshadweep": [{"name": "kavaratti", "lat": 10.57, "lon": 72.64}],
                      "andaman": [{"name": "port_blair", "lat": 11.60, "lon": 92.70}]}
    # constant Nino 3.4 for every year, enough to test the plumbing
    rows = "\n".join(f" {y} " + " ".join(["0.3"] * 12) for y in range(1950, 2027))
    table = build_all(cfg, log=lambda *_: None, nino_text=rows)
    assert set(table["region"]) == {"lakshadweep", "andaman"} and table["nino34"].notna().all()
    assert check_dhw(table, log=lambda *_: None) > 0.9          # own DHW agrees closely with "NOAA" DHW
    feat = add_features(table)
    assert "region" in feat.columns and feat["dhw"].notna().all()
    lab = add_labels(feat, 8)
    assert lab["label"].isin([0, 1]).all() and len(lab) > 500 and len(feature_columns(lab)) > 20
