"""
cv_utils.py — Cross-validation utilities shared across all training phases.
"""

import numpy as np
import pandas as pd


class PurgedTimeSeriesSplit:
    """
    k-fold time-series CV with a purge gap at each fold boundary.

    Prevents label leakage from overlapping forward-return windows (MAX_HOLD=5 bars).
    If `dates` array provided, purge is date-based (embargo_days calendar days);
    otherwise position-based with embargo=20 samples.
    """

    def __init__(self, n_splits: int = 5, embargo_days: int = 7):
        self.n_splits     = n_splits
        self.embargo_days = embargo_days

    def split(self, X, y=None, dates=None):
        n         = len(X)
        fold_size = n // (self.n_splits + 1)
        for fold in range(self.n_splits):
            val_start = fold_size * (fold + 1)
            val_end   = min(val_start + fold_size, n)

            if dates is not None:
                val_date    = pd.Timestamp(str(dates[val_start])[:10])
                cutoff      = val_date - pd.Timedelta(days=self.embargo_days)
                train_dates = pd.to_datetime([str(d)[:10] for d in dates[:val_start]])
                train_end   = int((train_dates <= cutoff).sum())
            else:
                train_end = max(0, val_start - 20)

            tr_idx  = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)
            if len(tr_idx) >= 100 and len(val_idx) >= 50:
                yield tr_idx, val_idx
