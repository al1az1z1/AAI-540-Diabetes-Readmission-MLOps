
import json
import boto3
from datetime import datetime, timezone


def publish_pipeline_metrics(
    evaluation_report_dict,
    region,
    namespace,
    pipeline_name
):
    """
    Publish model quality metrics from evaluation.json
    to Amazon CloudWatch.
    """

    cw = boto3.client(
        "cloudwatch",
        region_name=region
    )

    metrics = evaluation_report_dict["classification_metrics"]

    metric_data = []

    for metric_name in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ]:
        metric_data.append({
            "MetricName": metric_name.upper(),
            "Timestamp": datetime.now(timezone.utc),
            "Value": float(metrics[metric_name]["value"]),
            "Unit": "None",
            "Dimensions": [
                {
                    "Name": "Project",
                    "Value": "DiabetesReadmission"
                },
                {
                    "Name": "Pipeline",
                    "Value": pipeline_name
                }
            ]
        })

    cw.put_metric_data(
        Namespace=namespace,
        MetricData=metric_data
    )

    print("Published metrics to CloudWatch namespace:")
    print(namespace)


def create_pipeline_dashboard(
    region,
    namespace,
    pipeline_name,
    dashboard_name
):
    """
    Create a CloudWatch dashboard for pipeline model
    quality metrics.
    """

    cw = boto3.client(
        "cloudwatch",
        region_name=region
    )

    dashboard_body = {
        "widgets": [
            {
                "type": "metric",
                "x": 0,
                "y": 0,
                "width": 24,
                "height": 6,
                "properties": {
                    "metrics": [
                        [
                            namespace,
                            "ACCURACY",
                            "Project",
                            "DiabetesReadmission",
                            "Pipeline",
                            pipeline_name
                        ],
                        [
                            namespace,
                            "PRECISION",
                            "Project",
                            "DiabetesReadmission",
                            "Pipeline",
                            pipeline_name
                        ],
                        [
                            namespace,
                            "RECALL",
                            "Project",
                            "DiabetesReadmission",
                            "Pipeline",
                            pipeline_name
                        ],
                        [
                            namespace,
                            "F1",
                            "Project",
                            "DiabetesReadmission",
                            "Pipeline",
                            pipeline_name
                        ],
                        [
                            namespace,
                            "ROC_AUC",
                            "Project",
                            "DiabetesReadmission",
                            "Pipeline",
                            pipeline_name
                        ]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": region,
                    "title": "Diabetes Readmission Pipeline Model Quality Metrics",
                    "period": 300,
                    "stat": "Average"
                }
            }
        ]
    }

    cw.put_dashboard(
        DashboardName=dashboard_name,
        DashboardBody=json.dumps(dashboard_body)
    )

    print("CloudWatch dashboard created:")
    print(dashboard_name)
