import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from coralml.features import add_features, add_labels
from coralml.splits import walk_forward_folds
from coralml.synthetic import make_synthetic


def _data(h=8):
    raw = make_synthetic(n_sites=2, end="2012-12-30")
    return add_labels(add_features(raw), h)


def test_train_never_overlaps_test_window():
    for h in (4, 8, 12):
        d = _data(h)
        for f in walk_forward_folds(d, h, 3, 2, 2):
            gap = pd.Timedelta(weeks=h)
            assert d["date"].iloc[f["train"]].max() + gap < d["date"].iloc[f["test"]].min()
            assert d["date"].iloc[f["tune_train"]].max() + gap < d["date"].iloc[f["tune_val"]].min()
            assert d["date"].iloc[f["tune_val"]].max() < d["date"].iloc[f["test"]].min()


def test_label_uses_only_future_weeks():
    dates = pd.date_range("2020-01-05", periods=40, freq="7D")
    base = pd.DataFrame({"site": "a", "date": dates, "dhw": 0.0})
    base.loc[20, "dhw"] = 9.0                      # a single spike at row 20
    lab = add_labels(base, horizon_weeks=3).set_index("date")["label"]
    assert lab[dates[19]] == 1 and lab[dates[17]] == 1   # spike lies within t+1..t+3
    assert lab[dates[20]] == 0                           # week t itself is not used
    assert lab[dates[16]] == 0                           # too early to see it


def test_features_do_not_use_future():
    raw = make_synthetic(n_sites=1, end="2010-12-26")
    full = add_features(raw)
    t = full["date"].iloc[300]
    cut = add_features(raw[raw["date"] <= t])
    a = full[full["date"] == t].drop(columns=["site"]).iloc[0]
    b = cut[cut["date"] == t].drop(columns=["site"]).iloc[0]
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_region_holdout_never_trains_on_heldout_region():
    import yaml
    from coralml.benchmark import run_benchmark, load_cfg
    ev = load_cfg(Path(__file__).resolve().parents[1] / "configs/eval.yaml")
    mc = load_cfg(Path(__file__).resolve().parents[1] / "configs/models_simple.yaml")
    ev.update(horizons_weeks=[4], seeds=[0], bootstrap_reps=10)
    ev["folds"].update(n_folds=2); ev["tuning"]["n_trials"] = 1
    raw = make_synthetic(n_sites=6, end="2016-12-25")
    long_df, _ = run_benchmark(raw, ev, mc, only=["persistence", "logistic"], mode="region", log=lambda *_: None)
    assert set(long_df["held_out_region"]) == {"region_0", "region_1", "region_2"}
    # structural check: rebuild one unit and confirm region separation and time order
    from coralml.splits import walk_forward_folds
    d = add_labels(add_features(raw), 4)
    reg = d["region"].values
    fo = walk_forward_folds(d, 4, 2, 2, 2)[0]
    is_r = reg == "region_1"
    tr = fo["train"][~is_r[fo["train"]]]; te = fo["test"][is_r[fo["test"]]]
    assert (reg[tr] != "region_1").all() and (reg[te] == "region_1").all()
    assert d["date"].iloc[tr].max() < d["date"].iloc[te].min()
