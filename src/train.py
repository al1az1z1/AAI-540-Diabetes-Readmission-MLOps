# ============================================================
# 8. Create Training Script
# ============================================================
# Philosophy:
#
# Training logic should be separated from the
# orchestration notebook.
#
# This allows a data scientist to improve the model
# by editing train.py without changing the full
# SageMaker Pipeline definition.
#
# In this version, we use a stronger XGBoost
# champion-candidate configuration.
#
# Compared with V1, this version uses:
#
# - more boosting rounds
# - smaller learning rate
# - controlled tree depth
# - regularization
# - class imbalance weighting
#
# The goal is to create a better candidate model,
# while allowing the pipeline condition step to
# decide whether it should be registered.
# ============================================================


import os
import argparse
import pandas as pd
import xgboost as xgb


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_round", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--eta", type=float, default=0.03)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample_bytree", type=float, default=0.85)
    parser.add_argument("--min_child_weight", type=float, default=5)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--lambda_reg", type=float, default=2)

    args = parser.parse_args()

    train_path = "/opt/ml/input/data/train/train.csv"
    validation_path = "/opt/ml/input/data/validation/validation.csv"

    train_df = pd.read_csv(train_path, header=None)
    validation_df = pd.read_csv(validation_path, header=None)

    y_train = train_df.iloc[:, 0]
    X_train = train_df.iloc[:, 1:]

    y_validation = validation_df.iloc[:, 0]
    X_validation = validation_df.iloc[:, 1:]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    print("Training rows:", X_train.shape[0])
    print("Training features:", X_train.shape[1])
    print("Validation rows:", X_validation.shape[0])
    print("scale_pos_weight:", scale_pos_weight)

    dtrain = xgb.DMatrix(
        X_train,
        label=y_train
    )

    dvalidation = xgb.DMatrix(
        X_validation,
        label=y_validation
    )

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "gamma": args.gamma,
        "reg_lambda": args.lambda_reg,
        "scale_pos_weight": scale_pos_weight
    }

    evals = [
        (dtrain, "train"),
        (dvalidation, "validation")
    ]

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round,
        evals=evals,
        verbose_eval=25
    )

    model_dir = os.environ.get(
        "SM_MODEL_DIR",
        "/opt/ml/model"
    )

    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(
        model_dir,
        "xgboost-model"
    )

    model.save_model(model_path)

    print("Model saved to:")
    print(model_path)