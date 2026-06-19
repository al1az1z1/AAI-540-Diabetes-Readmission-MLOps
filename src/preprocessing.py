# ============================================================
# 7. Create Preprocessing Script
# ============================================================
# Philosophy:
#
# Preprocessing converts the raw diabetes dataset
# into machine-learning-ready datasets.
#
# This script is used inside SageMaker Pipelines as
# the Processing Step.
#
# It handles:
#
# - replacing "?" placeholders with true missing values
# - binary target conversion
# - feature selection
# - one-hot encoding
# - train / validation / test split
# - reserved production batch dataset creation
# - schema alignment for Batch Transform
#
# This is one of the most important scripts in the
# project because every downstream ML step depends
# on clean and consistent features.
# ============================================================


import argparse
import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import re

TARGET = "readmitted"
RANDOM_STATE = 42

FEATURES = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed",
    "insulin"
]

def clean_feature_name(col):
    col = str(col)
    col = re.sub(r"[^a-zA-Z0-9]", "_", col)
    col = re.sub(r"_+", "_", col)
    col = col.strip("_")

    if not re.match(r"^[a-zA-Z0-9]", col):
        col = "f_" + col

    return col[:64]



def label_first(df):
    """
    Move target column to the first position for
    SageMaker built-in XGBoost training.
    """

    return df[[TARGET] + [c for c in df.columns if c != TARGET]]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-size",
        type=int,
        default=15000
    )

    parser.add_argument(
        "--production-size",
        type=int,
        default=6000
    )

    args = parser.parse_args()

    base_dir = "/opt/ml/processing"

    input_path = os.path.join(
        base_dir,
        "input",
        "diabetic_data.csv"
    )

    df = pd.read_csv(input_path)

    # --------------------------------------------------------
    # 1. Replace missing-value placeholders
    # --------------------------------------------------------
    # The original dataset uses "?" to represent missing values.
    # We convert them to NaN before modeling.

    df = df.replace("?", np.nan)

    # --------------------------------------------------------
    # 2. Convert target to binary classification
    # --------------------------------------------------------
    # 1 = readmitted within 30 days
    # 0 = not readmitted within 30 days

    df[TARGET] = df[TARGET].replace({
        "<30": 1,
        ">30": 0,
        "NO": 0
    }).astype(int)

    # --------------------------------------------------------
    # 3. Create development sample and production reserve
    # --------------------------------------------------------

    df_sample, df_remaining = train_test_split(
        df,
        train_size=args.sample_size,
        stratify=df[TARGET],
        random_state=RANDOM_STATE
    )

    df_production_raw, _ = train_test_split(
        df_remaining,
        train_size=args.production_size,
        stratify=df_remaining[TARGET],
        random_state=RANDOM_STATE
    )

    df_sample = df_sample.reset_index(drop=True)
    df_production_raw = df_production_raw.reset_index(drop=True)

    # --------------------------------------------------------
    # 4. Build training/development feature matrix
    # --------------------------------------------------------

    df_model_raw = df_sample[FEATURES + [TARGET]].copy()

    X = df_model_raw.drop(columns=[TARGET])
    y = df_model_raw[TARGET]

    X_encoded = pd.get_dummies(
        X,
        drop_first=True
    )

    X_encoded.columns = [
        clean_feature_name(c)
        for c in X_encoded.columns
    ]

    feature_columns = X_encoded.columns.tolist()

    df_model = X_encoded.copy()
    df_model[TARGET] = y.values

    for col in df_model.columns:
        if df_model[col].dtype == "bool":
            df_model[col] = df_model[col].astype(int)

    df_model = df_model.apply(
        pd.to_numeric,
        errors="coerce"
    ).fillna(0)

    # --------------------------------------------------------
    # 5. Split development sample
    # --------------------------------------------------------
    # From the 15,000-row development sample:
    # - 80% train
    # - 10% validation
    # - 10% test

    train_df, temp_df = train_test_split(
        df_model,
        test_size=0.20,
        stratify=df_model[TARGET],
        random_state=RANDOM_STATE
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df[TARGET],
        random_state=RANDOM_STATE
    )

    train_out = label_first(train_df)
    val_out = label_first(val_df)
    test_out = label_first(test_df)

    # --------------------------------------------------------
    # 6. Build production batch data
    # --------------------------------------------------------

    prod_raw = df_production_raw[FEATURES + [TARGET]].copy()

    X_prod = prod_raw.drop(columns=[TARGET])
    y_prod = prod_raw[TARGET]

    X_prod_encoded = pd.get_dummies(
        X_prod,
        drop_first=True
    )

    X_prod_encoded.columns = [
        clean_feature_name(c)
        for c in X_prod_encoded.columns
    ]

    X_prod_encoded = X_prod_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    for col in X_prod_encoded.columns:
        if X_prod_encoded[col].dtype == "bool":
            X_prod_encoded[col] = X_prod_encoded[col].astype(int)

    X_prod_encoded = X_prod_encoded.apply(
        pd.to_numeric,
        errors="coerce"
    ).fillna(0)

    production_labeled = X_prod_encoded.copy()
    production_labeled[TARGET] = y_prod.values
    production_labeled = label_first(production_labeled)

    batch_features = X_prod_encoded.copy()

    # --------------------------------------------------------
    # 7. Save outputs for SageMaker Pipeline
    # --------------------------------------------------------

    os.makedirs(f"{base_dir}/train", exist_ok=True)
    os.makedirs(f"{base_dir}/validation", exist_ok=True)
    os.makedirs(f"{base_dir}/test", exist_ok=True)
    os.makedirs(f"{base_dir}/batch", exist_ok=True)
    os.makedirs(f"{base_dir}/monitoring", exist_ok=True)

    train_out.to_csv(
        f"{base_dir}/train/train.csv",
        header=False,
        index=False
    )

    val_out.to_csv(
        f"{base_dir}/validation/validation.csv",
        header=False,
        index=False
    )

    test_out.to_csv(
        f"{base_dir}/test/test.csv",
        header=False,
        index=False
    )

    batch_features.to_csv(
        f"{base_dir}/batch/batch_input.csv",
        header=False,
        index=False
    )

    production_labeled.to_csv(
        f"{base_dir}/monitoring/production_labeled.csv",
        header=True,
        index=False
    )

    schema = {
        "target": TARGET,
        "feature_columns": feature_columns,
        "train_rows": int(len(train_out)),
        "validation_rows": int(len(val_out)),
        "test_rows": int(len(test_out)),
        "production_rows": int(len(batch_features))
    }

    with open(f"{base_dir}/monitoring/schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    print("Preprocessing complete.")
    print(json.dumps(schema, indent=2))
