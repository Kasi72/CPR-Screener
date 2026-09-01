# [STANDALONE] One-off analysis script — not part of the ML training pipeline.
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report

df = pd.read_csv('xgb_v2_signals.csv')
df['pred_binary'] = (df['pred_ret'] > 0).astype(int)

auc_score = roc_auc_score(df['win'], df['pred_ret'])
auc_bin   = roc_auc_score(df['win'], df['pred_binary'])
acc       = (df['pred_binary'] == df['win']).mean()

print(f"AUC (pred_ret as score) : {auc_score:.4f}")
print(f"AUC (binary pred>0)     : {auc_bin:.4f}")
print(f"Direction accuracy       : {acc*100:.1f}%")
print()
print(classification_report(df['win'], df['pred_binary'], target_names=['Loss','Win']))

print("Per-rule AUC:")
for rid in [f'rule{i}' for i in range(1, 12)]:
    sub = df[df['rule'] == rid]
    if len(sub) < 20 or sub['win'].nunique() < 2:
        continue
    a = roc_auc_score(sub['win'], sub['pred_ret'])
    n = len(sub)
    wr = sub['win'].mean() * 100
    print(f"  {rid:<8} n={n:>5}  AUC={a:.4f}  WinRate={wr:.1f}%")
