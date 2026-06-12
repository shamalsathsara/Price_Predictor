"""
Train and save a sklearn Pipeline (preprocessing + estimator) for laptop price prediction.

This script:
- loads a CSV with laptop specifications and prices,
- normalizes common format variations (e.g. '8GB' -> 8, '1.37kg' -> 1.37),
- canonicalizes CPU/GPU/OS labels to a small set of values used during training,
- builds a ColumnTransformer to process numeric and categorical features,
- fits a RandomForestRegressor and saves the full Pipeline to disk.

Usage:
    python model/train_pipeline.py --data data/laptops.csv --out model/predictor.pickle
"""

import argparse
import logging
import os
import pickle
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_COLS = ['ram', 'weight', 'touchscreen', 'ips']
CATEGORICAL_COLS = ['company', 'typename', 'opsys', 'cpuname', 'gpuname']

def build_pipeline():
    """
    Create and return the sklearn Pipeline used for training and inference.
    We pass numeric features through as-is, and one-hot encode categorical ones.
    """
    # Handle unknown categories gracefully so the model doesn't crash on new data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', NUMERIC_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_COLS),
        ],
        remainder='drop',
        sparse_threshold=0,
    )

    # We use a Random Forest because it works great out-of-the-box for tabular data
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    return Pipeline([('pre', preprocessor), ('model', model)])


def clean_ram(ram_val):
    """
    Clean the RAM string (e.g. '8GB') and return an integer.
    """
    try:
        val_str = str(ram_val)
        return int(''.join(ch for ch in val_str if ch.isdigit()))
    except ValueError:
        return 0


def clean_weight(weight_val):
    """
    Clean the weight string (e.g. '1.37kg') and return a float.
    """
    try:
        val_str = str(weight_val).lower().replace('kg', '').strip()
        return float(val_str)
    except ValueError:
        return 0.0


def normalize_cpu(cpu_val):
    """
    Group various CPU names into a few primary categories to reduce complexity.
    """
    val_str = str(cpu_val).lower()
    if 'intel' in val_str and 'i7' in val_str:
        return 'Intel Core i7'
    if 'intel' in val_str and 'i5' in val_str:
        return 'Intel Core i5'
    if 'intel' in val_str and 'i3' in val_str:
        return 'Intel Core i3'
    if 'amd' in val_str:
        return 'AMD Processor'
    if 'intel' in val_str:
        return 'Other Intel Processor'
    return 'Other'


def normalize_gpu(gpu_val):
    """
    Group GPU names into main vendor categories.
    """
    val_str = str(gpu_val).lower()
    if 'nvidia' in val_str or 'geforce' in val_str:
        return 'NVIDIA GeForce'
    if 'intel' in val_str:
        return 'intel'
    if 'amd' in val_str:
        return 'amd'
    return 'Other'


def normalize_opsys(opsys_val):
    """
    Group Operating Systems into main categories.
    """
    val_str = str(opsys_val).lower()
    if 'windows' in val_str:
        return 'Windows'
    if 'mac' in val_str:
        return 'Mac'
    if 'linux' in val_str:
        return 'Linux'
    if 'no os' in val_str or val_str == 'no os':
        return 'No OS'
    return 'Other'


def load_and_clean_data(data_path, logger):
    """
    Load the CSV data and apply all our cleaning and normalization functions.
    """
    if not os.path.exists(data_path):
        logger.error("Training data not found: %s", data_path)
        sys.exit(1)

    # Some CSV files have weird encodings, so we fall back to latin-1 if utf-8 fails
    try:
        dataframe = pd.read_csv(data_path, encoding='utf-8')
    except UnicodeDecodeError:
        dataframe = pd.read_csv(data_path, encoding='latin-1')

    # Map different possible column names to our standard internal names
    col_map = {
        'Company': 'company',
        'TypeName': 'typename',
        'Ram': 'ram',
        'Weight': 'weight',
        'Cpu': 'cpuname',
        'Gpu': 'gpuname',
        'OpSys': 'opsys',
        'Price_euros': 'price'
    }
    dataframe = dataframe.rename(columns=col_map)

    required_cols = ['ram', 'weight', 'company', 'typename', 'opsys', 'cpuname', 'gpuname', 'price']
    missing_cols = [c for c in required_cols if c not in dataframe.columns]
    
    if missing_cols:
        logger.error("Missing required columns in CSV: %s", missing_cols)
        sys.exit(1)

    # Apply our cleaning functions
    dataframe['ram'] = dataframe['ram'].apply(clean_ram)
    dataframe['weight'] = dataframe['weight'].apply(clean_weight)
    dataframe['cpuname'] = dataframe['cpuname'].apply(normalize_cpu)
    dataframe['gpuname'] = dataframe['gpuname'].apply(normalize_gpu)
    dataframe['opsys'] = dataframe['opsys'].apply(normalize_opsys)

    # Fill in optional columns if they don't exist
    if 'touchscreen' not in dataframe.columns:
        dataframe['touchscreen'] = 0
    if 'ips' not in dataframe.columns:
        dataframe['ips'] = 0

    return dataframe


def train_model(args_parsed):
    """
    Main function to load data, train the model, and save the pipeline.
    """
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Load and prep our data
    dataframe = load_and_clean_data(args_parsed.data, logger)

    # Split features and target
    features = dataframe[NUMERIC_COLS + CATEGORICAL_COLS]
    target = dataframe['price']

    # Build the machine learning pipeline
    pipe = build_pipeline()

    # Split into train and test sets to evaluate performance
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Train the model!
    pipe.fit(features_train, target_train)

    # Check how well it did on the test set
    preds = pipe.predict(features_test)
    rmse = mean_squared_error(target_test, preds) ** 0.5
    logger.info("Validation RMSE: %.4f", rmse)

    # Ensure the output directory exists before saving
    out_dir = os.path.dirname(args_parsed.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Save our trained pipeline so the Flask app can use it later
    with open(args_parsed.out, 'wb') as file_out:
        pickle.dump(pipe, file_out)

    logger.info("Saved pipeline to %s", args_parsed.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrain laptop price predictor (pipeline)')
    parser.add_argument('--data', '-d', default='data/laptops.csv', help='Path to training CSV')
    parser.add_argument('--out', '-o', default='model/predictor.pickle', help='Output model path')
    cli_args = parser.parse_args()
    train_model(cli_args)
