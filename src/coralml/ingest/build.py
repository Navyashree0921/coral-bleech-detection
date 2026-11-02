"""Combine CRW weekly site series + Niño 3.4 into the table the benchmark expects."""
import numpy as np
import pandas as pd
from .crw import find_water_pixel, download_site, to_weekly
from .nino import parse_psl_standard, download, join_weekly


def site_table(weekly, site, region):
    return pd.DataFrame({"site": site, "region": region, "date": weekly.index,
                         "sst": weekly["CRW_SST"].values,
                         "mmm": (weekly["CRW_SST"] - weekly["CRW_HOTSPOT"]).values,   # HotSpot = SST - MMM
                         "dhw": weekly["CRW_DHW"].values})


def attach_nino(table, monthly, lag_months):
    out = table.copy()
    out["nino34"] = join_weekly(out["date"], monthly, lag_months)
    return out.dropna(subset=["nino34", "dhw", "sst", "mmm"]).reset_index(drop=True)


def check_dhw(table, log=print):
    """Sanity check: rebuild DHW from weekly hotspots and compare with NOAA's own value."""
    d = table.sort_values(["site", "date"]).copy()
    hs = (d["sst"] - d["mmm"]).where(lambda s: s >= 1.0, 0.0)
    mine = hs.groupby(d["site"]).transform(lambda s: s.rolling(12, min_periods=12).sum())
    ok = mine.notna()
    corr = float(np.corrcoef(mine[ok], d.loc[ok, "dhw"])[0, 1]) if ok.sum() > 10 else float("nan")
    log(f"DHW check: correlation(own weekly DHW, NOAA DHW) = {corr:.3f}, "
        f"mean abs diff = {float((mine[ok]-d.loc[ok,'dhw']).abs().mean()):.3f} C-weeks")
    return corr


def build_all(cfg, log=print, nino_text=None):
    monthly = parse_psl_standard(nino_text or download(cfg["nino"]["url"]))
    tables = []
    for region, sites in cfg["regions"].items():
        for s in sites:
            log(f"{region}/{s['name']}")
            lat, lon = find_water_pixel(cfg["crw"], s["lat"], s["lon"], log)
            weekly = to_weekly(download_site(cfg["crw"], lat, lon, log))
            tables.append(site_table(weekly, s["name"], region))
    table = attach_nino(pd.concat(tables, ignore_index=True), monthly, cfg["nino"]["lag_months"])
    check_dhw(table, log)
    return table
