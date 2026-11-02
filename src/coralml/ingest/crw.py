"""Download NOAA Coral Reef Watch 5 km daily series for a point via ERDDAP (griddap, CSV)."""
import io
import time
import numpy as np
import pandas as pd
import requests

STEP = 0.05  # grid spacing in degrees (5 km)


def _snap(v):
    """Grid centres sit at multiples of 0.05 offset by 0.025."""
    return round((np.floor(v / STEP) + 0.5) * STEP, 3)


def build_url(base, variables, t0, t1, lat, lon):
    sel = f"[({t0}T12:00:00Z):1:({t1}T12:00:00Z)][({lat})][({lon})]"
    return base + "?" + ",".join(v + sel for v in variables)


def parse_erddap_csv(text):
    """ERDDAP CSV: row 1 = names, row 2 = units. Returns daily frame indexed by date."""
    df = pd.read_csv(io.StringIO(text), skiprows=[1])
    df["date"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None).dt.normalize()
    return df.drop(columns=[c for c in ("time", "latitude", "longitude") if c in df.columns]).set_index("date")


def _get(url, tries=3, timeout=180):
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.text
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:
            last = str(e)
        time.sleep(2 * (k + 1))
    raise RuntimeError(f"ERDDAP request failed: {last}\n{url}")


def is_water(cfg, lat, lon):
    """Probe one known-good date; land / masked pixels come back empty."""
    url = build_url(cfg["erddap_url"], ["CRW_SST"], "2020-06-01", "2020-06-03", lat, lon)
    try:
        df = parse_erddap_csv(_get(url, tries=1, timeout=60))
    except RuntimeError:
        return False
    return bool(df["CRW_SST"].notna().any())


def find_water_pixel(cfg, lat, lon, log=print):
    r = _snap(lat), _snap(lon)
    n = int(cfg.get("water_search_steps", 3))
    cands = sorted(((di, dj) for di in range(-n, n + 1) for dj in range(-n, n + 1)),
                   key=lambda p: (p[0] ** 2 + p[1] ** 2))
    for di, dj in cands:
        la, lo = round(r[0] + di * STEP, 3), round(r[1] + dj * STEP, 3)
        if is_water(cfg, la, lo):
            if (di, dj) != (0, 0):
                log(f"  point ({lat},{lon}) is land/masked; using water pixel ({la},{lo}), {abs(di)+abs(dj)} pixel(s) away")
            return la, lo
    raise RuntimeError(f"no water pixel within {n} pixels of ({lat}, {lon})")


def download_site(cfg, lat, lon, log=print):
    base, vars_ = cfg["erddap_url"], cfg["variables"]
    start, end = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
    edges = [start]
    while True:
        nxt = edges[-1] + pd.DateOffset(years=int(cfg["chunk_years"]))
        if nxt > end:
            break
        edges.append(nxt)
    parts = []
    for i, a in enumerate(edges):
        b = (edges[i + 1] - pd.Timedelta(days=1)) if i + 1 < len(edges) else end
        url = build_url(base, vars_, a.date(), b.date(), lat, lon)
        parts.append(parse_erddap_csv(_get(url)))
        log(f"  {a.date()} to {b.date()}: {len(parts[-1])} days")
    out = pd.concat(parts).sort_index()
    return out[~out.index.duplicated()]


def to_weekly(daily):
    """Weekly table (weeks end on Sunday): the last valid daily value in each week."""
    return daily.resample("W-SUN").last().dropna(subset=["CRW_SST"])
