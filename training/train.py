#!/usr/bin/env python3
"""
LightGBM-based imbalanced binary classifier trainer.
- Expects a preprocessed patient-year table (or this script can do minimal preprocessing).
- Uses MLflow to log params/metrics/artifacts.
- Selects threshold on validation by maximizing F1 or by recall target.
"""

import os
import json
import joblib
import yaml
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, precision_score, recall_score,
                             f1_score, brier_score_loss, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

from data_prep import prepare_data

# ---------------- helpers ----------------
def now_tag():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def select_threshold(y_true, y_probs, method='f1_max', recall_target=None):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
    # thresholds length = len(precision)-1
    if method == 'f1_max':
        idx = np.nanargmax(f1_scores)
        # if idx corresponds to prec/recl/threshold mapping: adjust index if necessary
        # f1_scores aligned with precision[:-1] because precision,recall arrays len = n_thresholds+1
        if idx >= len(thresholds):
            idx = len(thresholds)-1
        return thresholds[idx], float(f1_scores[idx])
    if method == 'recall_target' and recall_target is not None:
        idxs = np.where(recall[:-1] >= recall_target)[0]  # drop last element
        if len(idxs) == 0:
            # fallback to min threshold (most permissive)
            return thresholds[-1], float(f1_scores[np.nanargmax(f1_scores)])
        chosen = idxs[0]
        if chosen >= len(thresholds):
            chosen = len(thresholds)-1
        return thresholds[chosen], float(f1_scores[chosen])
    # fallback
    idx = np.nanargmax(f1_scores)
    if idx >= len(thresholds):
        idx = len(thresholds)-1
    return thresholds[idx], float(f1_scores[idx])

def evaluate_all(y_true, y_probs, threshold):
    y_pred = (y_probs >= threshold).astype(int)
    metrics = {}
    metrics['roc_auc'] = float(roc_auc_score(y_true, y_probs))
    metrics['pr_auc'] = float(average_precision_score(y_true, y_probs))
    metrics['precision'] = float(precision_score(y_true, y_pred))
    metrics['recall'] = float(recall_score(y_true, y_pred))
    metrics['f1'] = float(f1_score(y_true, y_pred))
    metrics['brier'] = float(brier_score_loss(y_true, y_probs))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)})
    return metrics

def plot_pr_curve(y_true, y_probs, outpath):
    prec, rec, _ = precision_recall_curve(y_true, y_probs)
    plt.figure(figsize=(6,5))
    plt.plot(rec, prec, label='PR curve')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve')
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(outpath); plt.close()

def plot_roc_curve(y_true, y_probs, outpath):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    aucv = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'ROC AUC={aucv:.3f}')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve')
    plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(outpath); plt.close()

# ---------------- main ----------------

with open("config.yaml", 'r') as f:
    cfg = yaml.safe_load(f)

data_path = cfg['data']['path']
use_dask = cfg['data'].get('use_dask', True)
X_train, y_train, X_val, y_val, X_test, y_test, preproc = prepare_data(data_path, use_dask, cfg)

out_dir = Path(cfg['artifact']['out_dir']) 
print("out_dir:", out_dir)
ensure_dir(out_dir)
joblib.dump(preproc, out_dir / "preprocessor.joblib")

# LightGBM dataset and scale_pos_weight for imbalance
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
if pos == 0:
    raise ValueError("No positive samples in training set!")
scale_pos_weight = neg / pos
print(f"Train pos/neg = {pos}/{neg}, scale_pos_weight = {scale_pos_weight:.3f}")

lgb_params = cfg['lgb_params'].copy()
# attach imbalance handling
if cfg['lgb_params'].get('is_unbalance', False) is False:
    # prefer scale_pos_weight if provided
    lgb_params['scale_pos_weight'] = cfg['lgb_params'].get('scale_pos_weight', scale_pos_weight)
# ensure objective = binary
lgb_params['objective'] = 'binary'
lgb_params['verbosity'] = -1
lgb_params['metric'] = 'None'  # we will compute ourselves

# Prepare datasets for early stopping
dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

# Start MLflow experiment
mlflow.set_tracking_uri( os.getenv("MLFLOW_TRACKING_URI", "file:/mlruns"))
mlflow.set_experiment(cfg.get('mlflow_experiment', 'lgbm_imbalanced_experiment'))

with mlflow.start_run():

    # Log hyperparameters
    mlflow.log_params(lgb_params)
    mlflow.log_param('scale_pos_weight', float(scale_pos_weight))
    mlflow.log_param('n_train', int(len(y_train)))
    mlflow.log_param('n_val', int(len(y_val)))
    mlflow.log_param('n_test', int(len(y_test)))

    # Train with early stopping
    print("Training LightGBM...")
    bst = lgb.train(
        # params=lgb_params,
        params={**lgb_params, "metric": "auc"},
        train_set=dtrain,
        num_boost_round=cfg['num_boost_round'],
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        callbacks=[
    early_stopping(stopping_rounds=cfg.get('early_stopping_rounds', 50)),
    log_evaluation(period=cfg.get('verbose_eval', 50))
      ]

    )

    # Save model artifact
    model_path = out_dir / "model.txt"
    bst.save_model(str(model_path))
    mlflow.log_artifact(str(model_path), artifact_path='model')
    joblib.dump(bst, out_dir / "lgbm_model.joblib")

    # Predict probabilities on val & test
    val_probs = bst.predict(X_val, num_iteration=bst.best_iteration)
    test_probs = bst.predict(X_test, num_iteration=bst.best_iteration)

    # Select threshold using validation set
    thr, thr_f1 = select_threshold(y_val, val_probs, method=cfg.get('threshold_method', 'f1_max'), recall_target=cfg.get('recall_target'))
    mlflow.log_metric('val_selected_threshold', float(thr))
    mlflow.log_metric('val_threshold_f1', float(thr_f1))
    print(f"Selected threshold: {thr:.4f}, val_f1: {thr_f1:.4f}")

    # Evaluate on test
    test_metrics = evaluate_all(y_test, test_probs, thr)
    for k,v in test_metrics.items():
        mlflow.log_metric(f"test_{k}", float(v))

    # Save metrics JSON
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({'val_threshold_f1': thr_f1, 'threshold': thr, 'test_metrics': test_metrics}, f, indent=2)
    mlflow.log_artifact(str(out_dir / "metrics.json"), artifact_path='metrics')

    # PR & ROC plots
    pr_path = out_dir / "pr_curve.png"
    roc_path = out_dir / "roc_curve.png"
    plot_pr_curve(y_test, test_probs, str(pr_path))
    plot_roc_curve(y_test, test_probs, str(roc_path))
    mlflow.log_artifact(str(pr_path), artifact_path='plots')
    mlflow.log_artifact(str(roc_path), artifact_path='plots')

    # Precision@k (e.g., top 1%, top 5%)
    for k in cfg.get('precision_at_k', [0.01, 0.05]):
        cutoff = int(np.ceil(len(test_probs) * k))
        idx = np.argsort(-test_probs)[:cutoff]
        prec_at_k = y_test[idx].sum() / max(1, len(idx))
        mlflow.log_metric(f'precision_at_{int(k*100)}pct', float(prec_at_k))

    # Save a small feature importance table (gain)
    try:
        fmap = bst.feature_name()
        fimp = bst.feature_importance(importance_type='gain')
        imp_df = pd.DataFrame({'feature': fmap, 'gain': fimp})
        imp_csv = out_dir / "feature_importance.csv"
        imp_df.sort_values('gain', ascending=False).to_csv(imp_csv, index=False)
        mlflow.log_artifact(str(imp_csv), artifact_path='feature_importance')
    except Exception as e:
        print("Failed to log feature importance:", e)

    # Log config
    cfg_out = out_dir / "config_used.yaml"
    with open(cfg_out, 'w') as f:
        yaml.safe_dump(cfg, f)
    mlflow.log_artifact(str(cfg_out), artifact_path='config')

    print("Run finished. Artifacts at:", out_dir)
