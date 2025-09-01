"""
Prepare data for train, validation and test
"""
import os
import json
import time
import joblib
import yaml
import click
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# Try to import dask for scalable ops
USE_DASK = True
try:
    import dask.dataframe as dd
except Exception:
    USE_DASK = False

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, precision_score, recall_score,
                             f1_score, brier_score_loss, confusion_matrix)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


# Utilities
def now_tag():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


# Ingest data (Dask preferred)
def read_table(path, use_dask=True, config=None):
  if use_dask and USE_DASK:
      print("Reading with Dask...")
      df = dd.read_csv(path, assume_missing=True) if str(path).endswith(".csv") or str(path).endswith(".csv.gz") else dd.read_parquet(path)
      return df
  else:
      print("Reading with pandas chunked...")
      return None  # fallback will be handled in pipeline


# Temporal aggregations (previous X years)
def compute_prev_year_aggregations(df, cfg):
  """
  Expects a patient-year table with columns ['patient_id', 'calendar_year', ...]
  Computes rolling aggregations over previous X years (not including current year).
  For Dask: uses groupby-apply pattern (may require partitions).
  """
  prev_years = cfg['temporal']['prev_years']
  gid = cfg['data']['groupby_id'] # Correctly access groupby_id from data section
  year_col = cfg['data']['year_col']

  # For scalability, we assume data is sorted by (patient_id, calendar_year)
  # Strategy:
  #  - For each patient, create a rolling window over previous X rows by year.
  #  - Compute aggregations (mean, sum, last) for selected numeric features.
  # Because Dask groupby-apply can be heavy, we implement a chunked pandas approach
  # when df is concrete (pd.DataFrame).
  #
  # We'll expose a helper to compute rolling features given a pandas dataframe.
  numeric_cols = [c for c in df.columns if df[c].dtype.kind in 'fi' and c not in [cfg['data']['target_col'], year_col]]
  agg_fns = ['mean', 'sum', 'max']

  def patient_rolling(group):
      group = group.sort_values(year_col)
      for col in numeric_cols:
          # previous X years: shift cumulative window by 1..X
          rolled = group[col].shift(1).rolling(window=prev_years, min_periods=0).agg(['mean', 'sum', 'max'])
          rolled.columns = [f"{col}_prev{prev_years}_mean", f"{col}_prev{prev_years}_sum", f"{col}_prev{prev_years}_max"]
          group = pd.concat([group, rolled], axis=1)
      return group

  # If Dask DataFrame passed, convert sample or implement map_partitions with groupby
  if USE_DASK and isinstance(df, dd.DataFrame):
      # Danger: groupby-apply over many partitions may shuffle; recommend calling compute on groups per partition
      print("Computing rolling aggregations with Dask (map_partitions).")
      # A safer pattern: persist, repartition by patient id, then map_partitions
      df2 = df.map_partitions(lambda pdf: pdf.sort_values([gid, year_col]))
      # Convert to pandas partitions and apply groupby.apply
      # Note: for very large data, consider writing a custom aggregation in chunked SQL or Spark.
      return df2.map_partitions(lambda pdf: pdf.groupby(gid).apply(patient_rolling)).reset_index(drop=True)
  else:
      # pandas DataFrame: chunked grouping to avoid huge memory
      print("Computing rolling aggregations with pandas (groupby apply).")
      return df.groupby(gid, group_keys=False).apply(patient_rolling).reset_index(drop=True)


# Preprocessing pipeline builder
def build_preprocessor(df_sample, cfg):
  # gather column types
  target = cfg['data']['target_col']
  year_col = cfg['data']['year_col'] # Access year_col from cfg

  # Identify categorical and numeric columns
  categorical_cols = [c for c in df_sample.columns if df_sample[c].dtype == 'object' or df_sample[c].dtype.name == 'category']
  numeric_cols = [c for c in df_sample.columns if df_sample[c].dtype.kind in 'fi' and c not in categorical_cols + [target, year_col, cfg['data']['groupby_id']]] # Exclude groupby_id

  # Build sklearn ColumnTransformer
  numeric_transformers = [('scaler', StandardScaler())]
  cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # Use sparse=False for dense output

  preproc = ColumnTransformer(transformers=[
      ('num', StandardScaler(), numeric_cols),
      ('cat', cat_transformer, categorical_cols)
  ], remainder='drop')
  return preproc, numeric_cols, categorical_cols



# Prepare train, test and validation data
def prepare_data(data_path, use_dask, cfg):

  # Read table (Dask preferred)
  ddf = read_table(data_path, use_dask=use_dask, config=cfg)
  if ddf is None:
      # pandas fallback: read in chunks (CSV)
      print("Pandas fallback: streaming CSV")
      chunks = []
      chunksize = cfg['data'].get('chunksize', 200000)
      for chunk in pd.read_csv(data_path, compression='infer', chunksize=chunksize):
          chunks.append(chunk)
      df = pd.concat(chunks, ignore_index=True)
      del chunks
  else:
      # Convert Dask to pandas sample for schema, but avoid compute whole
      print("Sampling data for schema and small transforms...")
      sample = ddf.sample(frac=0.001).compute() if USE_DASK else ddf.head(1000)
      # For simplicity, compute entire dataset to pandas for flows requiring .groupby.apply
      print("Materializing dataframe from Dask for group-based temporal aggregations (may be memory heavy).")
      # df = ddf.compute()
      df = ddf

  print("Initial rows:", len(df))

  # Filter/ensure types
  target = cfg['data']['target_col']
  year_col = cfg['data']['year_col']
  df[year_col] = df[year_col].astype(int)
  df[target] = df[target].astype(int)

  # Compute previous-X-year aggregations (per-patient)
  print("Computing temporal aggregations...")
  df = compute_prev_year_aggregations(df, cfg)
  print("After temporal features:", df.shape)
  df = ddf.compute()

  # Standard preprocessing: impute, encode, clip — simple approach
  # Fill missing
  numeric_cols = [c for c in df.columns if df[c].dtype.kind in 'fi' and c not in [target, year_col, cfg['data']['groupby_id']]] # Exclude groupby_id
  cat_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].dtype.name == 'category']

  for c in numeric_cols:
      df[c] = df[c].fillna(df[c].median())
  for c in cat_cols:
      df[c] = df[c].fillna('missing')

  # Split train/val/test by patient to avoid leakage
  patients = df['patient_id'].unique()
  train_p, temp_p = train_test_split(patients, test_size=0.3, random_state=42)
  val_p, test_p = train_test_split(temp_p, test_size=0.5, random_state=42)

  train_df = df[df['patient_id'].isin(train_p)].reset_index(drop=True)
  val_df = df[df['patient_id'].isin(val_p)].reset_index(drop=True)
  test_df = df[df['patient_id'].isin(test_p)].reset_index(drop=True)

  print("Split sizes: train={}, val={}, test={}".format(len(train_df), len(val_df), len(test_df)))

  # Build preprocessor from training sample
  preproc, num_cols, cat_cols = build_preprocessor(train_df.sample(min(10000, len(train_df))), cfg=cfg) # Pass cfg to build_preprocessor
  print("Built preprocessor.")

  # Fit encoders/scalers on train
  X_train = train_df.drop(columns=[target])
  y_train = train_df[target].values
  X_val = val_df.drop(columns=[target])
  y_val = val_df[target].values
  X_test = test_df.drop(columns=[target])
  y_test = test_df[target].values

  # Fit preprocessor (ColumnTransformer)
  print("Fitting preprocessor on training data (may be memory heavy) ...")
  preproc.fit(X_train)

  # Transform datasets (dense arrays)
  X_train_t = preproc.transform(X_train)
  X_val_t = preproc.transform(X_val)
  X_test_t = preproc.transform(X_test)
  print("Transformed feature shapes:", X_train_t.shape, X_val_t.shape, X_test_t.shape)

  X_val_t.to_csv("X_val.csv.gz", index=False, compression="gzip")
  y_val.to_csv("y_val.csv.gz", index=False, compression="gzip")

  return X_train_t, y_train, X_val_t, y_val, X_test_t, y_test, preproc


