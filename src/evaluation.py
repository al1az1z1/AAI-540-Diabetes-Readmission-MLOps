# ============================================================
# 9. Create Evaluation Script
# ============================================================
# Philosophy:
#
# Evaluation is separated from training because
# model validation should be an independent step.
#
# This script loads the trained XGBoost model,
# evaluates it on the test dataset, and writes an
# evaluation.json file.
#
# SageMaker Pipelines reads this evaluation report
# in the Condition Step.
#
# If the F1-score meets the threshold, the model is
# registered and used for Batch Transform.
#
# If the F1-score does not meet the threshold, the
# pipeline fails safely.
# ============================================================


import os
import json
import tarfile
import pathlib
import argparse

import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prediction-threshold",
        type=float,
        default=0.50
    )

    args = parser.parse_args()

    model_tar_path = "/opt/ml/processing/model/model.tar.gz"

    with tarfile.open(model_tar_path) as tar:
        tar.extractall(path="/opt/ml/processing/model")

    model_file = "/opt/ml/processing/model/xgboost-model"

    booster = xgb.Booster()
    booster.load_model(model_file)

    test_path = "/opt/ml/processing/test/test.csv"

    df = pd.read_csv(
        test_path,
        header=None
    )

    y_test = df.iloc[:, 0].astype(int)
    X_test = df.iloc[:, 1:]

    dtest = xgb.DMatrix(X_test)

    probabilities = booster.predict(dtest)

    predictions = (
        probabilities >= args.prediction_threshold
    ).astype(int)

    report = {
        "classification_metrics": {
            "accuracy": {
                "value": float(
                    accuracy_score(y_test, predictions)
                )
            },
            "precision": {
                "value": float(
                    precision_score(
                        y_test,
                        predictions,
                        zero_division=0
                    )
                )
            },
            "recall": {
                "value": float(
                    recall_score(
                        y_test,
                        predictions,
                        zero_division=0
                    )
                )
            },
            "f1": {
                "value": float(
                    f1_score(
                        y_test,
                        predictions,
                        zero_division=0
                    )
                )
            },
            "roc_auc": {
                "value": float(
                    roc_auc_score(
                        y_test,
                        probabilities
                    )
                )
            },
            "confusion_matrix": {
                "value": confusion_matrix(
                    y_test,
                    predictions
                ).tolist()
            },
            "prediction_threshold": {
                "value": float(
                    args.prediction_threshold
                )
            }
        }
    }

    output_dir = "/opt/ml/processing/evaluation"

    pathlib.Path(output_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    evaluation_path = f"{output_dir}/evaluation.json"

    with open(evaluation_path, "w") as f:
        json.dump(
            report,
            f,
            indent=2
        )

    print(json.dumps(report, indent=2))
