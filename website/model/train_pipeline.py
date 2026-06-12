"""
Train and save a sklearn Pipeline that includes preprocessing (ColumnTransformer)
and a RandomForestRegressor. The pipeline will be saved to `model/predictor.pickle`.

Usage:
    python model/train_pipeline.py --data path/to/train.csv --out model/predictor.pickle

CSV expected columns: `ram,weight,company,typename,opsys,cpuname,gpuname,price`.
Optional: `touchscreen,ips` (0/1).
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Define columns
NUMERIC_COLS = ['ram', 'weight', 'touchscreen', 'ips']
CATEGORICAL_COLS = ['company', 'typename', 'opsys', 'cpuname', 'gpuname']


def build_pipeline():
    # OneHotEncoder will handle unknown categories by ignoring them at transform time
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', NUMERIC_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_COLS),
        ],
        remainder='drop',
        sparse_threshold=0,
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    pipe = Pipeline([('pre', preprocessor), ('model', model)])
    return pipe


def main(args):
    if not os.path.exists(args.data):
        print(f"Training data not found: {args.data}")
        sys.exit(1)

    # try utf-8 first, fall back to latin-1 for files with special characters
    try:
        df = pd.read_csv(args.data, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(args.data, encoding='latin-1')
    # Normalize column names from common laptop CSV formats
    # Map possible source columns to the expected names
    col_map = {}
    if 'Company' in df.columns:
        col_map['Company'] = 'company'
    if 'TypeName' in df.columns:
        col_map['TypeName'] = 'typename'
    if 'Ram' in df.columns:
        col_map['Ram'] = 'ram'
    if 'Weight' in df.columns:
        col_map['Weight'] = 'weight'
    if 'Cpu' in df.columns:
        col_map['Cpu'] = 'cpuname'
    if 'Gpu' in df.columns:
        col_map['Gpu'] = 'gpuname'
    if 'OpSys' in df.columns:
        col_map['OpSys'] = 'opsys'
    if 'Price_euros' in df.columns:
        col_map['Price_euros'] = 'price'

    df = df.rename(columns=col_map)

    # Check required columns now
    required = ['ram', 'weight', 'company', 'typename', 'opsys', 'cpuname', 'gpuname', 'price']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing required columns in CSV: {missing}")
        sys.exit(1)

    # clean and normalize columns
    # Ram: strings like '8GB' -> 8
    def clean_ram(x):
        try:
            s = str(x)
            return int(''.join(ch for ch in s if ch.isdigit()))
        except Exception:
            return 0

    # Weight: strings like '1.37kg' -> 1.37
    def clean_weight(x):
        try:
            s = str(x).lower().replace('kg','').strip()
            return float(s)
        except Exception:
            return 0.0

    def normalize_cpu(x):
        s = str(x).lower()
        if 'intel' in s and 'i7' in s:
            return 'Intel Core i7'
        if 'intel' in s and 'i5' in s:
            return 'Intel Core i5'
        if 'intel' in s and 'i3' in s:
            return 'Intel Core i3'
        if 'amd' in s:
            return 'AMD Processor'
        if 'intel' in s:
            return 'Other Intel Processor'
        return 'Other'

    def normalize_gpu(x):
        s = str(x).lower()
        if 'nvidia' in s or 'geforce' in s:
            return 'NVIDIA GeForce'
        if 'intel' in s:
            return 'intel'
        if 'amd' in s:
            return 'amd'
        return 'Other'

    def normalize_opsys(x):
        s = str(x).lower()
        if 'windows' in s:
            return 'Windows'
        if 'mac' in s:
            return 'Mac'
        if 'linux' in s:
            return 'Linux'
        if 'no os' in s or 'no os' == s:
            return 'No OS'
        return 'Other'

    df['ram'] = df['ram'].apply(clean_ram)
    df['weight'] = df['weight'].apply(clean_weight)
    df['cpuname'] = df['cpuname'].apply(normalize_cpu)
    df['gpuname'] = df['gpuname'].apply(normalize_gpu)
    df['opsys'] = df['opsys'].apply(normalize_opsys)

    # ensure optional columns
    if 'touchscreen' not in df.columns:
        df['touchscreen'] = 0
    if 'ips' not in df.columns:
        df['ips'] = 0

    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df['price']

    pipe = build_pipeline()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    # mean_squared_error no longer accepts `squared` in this sklearn version
    rmse = mean_squared_error(y_test, preds) ** 0.5
    print(f"Validation RMSE: {rmse:.4f}")

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, 'wb') as f:
        pickle.dump(pipe, f)

    print(f"Saved pipeline to {args.out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrain laptop price predictor (pipeline)')
    parser.add_argument('--data', '-d', default='data/laptops.csv', help='Path to training CSV')
    parser.add_argument('--out', '-o', default='model/predictor.pickle', help='Output model path')
    args = parser.parse_args()
    main(args)
