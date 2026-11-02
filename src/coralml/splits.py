"""Walk-forward splits by date with a gap so labels never peek into the test period."""
import numpy as np
import pandas as pd


def walk_forward_folds(df, horizon_weeks, n_folds, test_block_years, inner_val_years):
    years = np.sort(df["year"].unique())
    last = years.max()
    gap = pd.Timedelta(weeks=horizon_weeks)
    folds = []
    for k in range(n_folds):
        t_end = last - (n_folds - 1 - k) * test_block_years
        t_start = t_end - test_block_years + 1
        test_start = pd.Timestamp(year=int(t_start), month=1, day=1)
        test_end = pd.Timestamp(year=int(t_end) + 1, month=1, day=1)
        test = (df["date"] >= test_start) & (df["date"] < test_end)
        train = df["date"] < (test_start - gap)
        val_start = test_start - pd.DateOffset(years=inner_val_years)
        tune_val = (df["date"] >= val_start) & (df["date"] < (test_start - gap))
        tune_train = df["date"] < (val_start - gap)
        folds.append(dict(fold=k, test_years=(int(t_start), int(t_end)),
                          train=np.where(train)[0], test=np.where(test)[0],
                          tune_train=np.where(tune_train)[0], tune_val=np.where(tune_val)[0]))
    return folds
