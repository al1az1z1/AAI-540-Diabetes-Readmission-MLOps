
import time
import boto3
import pandas as pd


def create_or_reuse_feature_group(
    df_features,
    bucket_name,
    region,
    role,
    feature_group_name,
    run_ingestion=False
):
    """
    Create or reuse a SageMaker Feature Group.
    Optionally ingest engineered feature records.
    """

    sm_client = boto3.client(
        "sagemaker",
        region_name=region
    )

    featurestore_runtime = boto3.client(
        "sagemaker-featurestore-runtime",
        region_name=region
    )

    df_features = df_features.copy()

    if "record_id" not in df_features.columns:
        df_features["record_id"] = df_features.index.astype(str)

    if "event_time" not in df_features.columns:
        df_features["event_time"] = pd.Timestamp.utcnow().strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    for col in df_features.columns:
        if df_features[col].dtype == "bool":
            df_features[col] = df_features[col].astype(int)

    feature_definitions = []

    for col in df_features.columns:

        if col == "event_time":
            feature_type = "String"

        elif pd.api.types.is_integer_dtype(df_features[col]):
            feature_type = "Integral"

        elif pd.api.types.is_float_dtype(df_features[col]):
            feature_type = "Fractional"

        else:
            feature_type = "String"

        feature_definitions.append({
            "FeatureName": col,
            "FeatureType": feature_type
        })

    try:
        sm_client.describe_feature_group(
            FeatureGroupName=feature_group_name
        )

        print("Feature Group already exists:")
        print(feature_group_name)

    except sm_client.exceptions.ResourceNotFound:

        sm_client.create_feature_group(
            FeatureGroupName=feature_group_name,
            RecordIdentifierFeatureName="record_id",
            EventTimeFeatureName="event_time",
            FeatureDefinitions=feature_definitions,
            OfflineStoreConfig={
                "S3StorageConfig": {
                    "S3Uri": f"s3://{bucket_name}/feature-store/"
                },
                "DisableGlueTableCreation": False
            },
            RoleArn=role
        )

        print("Creating Feature Group:")
        print(feature_group_name)

    while True:

        response = sm_client.describe_feature_group(
            FeatureGroupName=feature_group_name
        )

        status = response["FeatureGroupStatus"]

        print("Feature Group status:", status)

        if status == "Created":
            break

        if status in ["CreateFailed", "DeleteFailed"]:
            raise Exception(
                f"Feature Group failed with status: {status}"
            )

        time.sleep(30)

    if run_ingestion:

        for i, row in df_features.iterrows():

            record = []

            for col, value in row.items():

                if pd.isna(value):
                    value = ""

                record.append({
                    "FeatureName": col,
                    "ValueAsString": str(value)
                })

            featurestore_runtime.put_record(
                FeatureGroupName=feature_group_name,
                Record=record
            )

            if i % 1000 == 0:
                print(f"Ingested {i} records...")

        print("Feature data ingested into SageMaker Feature Store.")

    else:

        print("Skipping ingestion because run_ingestion = False.")

    print("Feature Group Name:", feature_group_name)
    print("Offline Store Path:", f"s3://{bucket_name}/feature-store/")

    return feature_group_name
