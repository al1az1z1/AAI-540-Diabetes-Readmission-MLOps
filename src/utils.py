
import re
import time
import boto3


def clean_feature_name(col):
    """
    Convert feature names into a SageMaker-safe format.
    """

    col = str(col)

    col = re.sub(r"[^a-zA-Z0-9]", "_", col)
    col = re.sub(r"_+", "_", col)
    col = col.strip("_")

    if not re.match(r"^[a-zA-Z0-9]", col):
        col = "f_" + col

    return col[:64]


def create_or_reuse_bucket(bucket_name, region):
    """
    Create an S3 bucket if it does not exist.
    Reuse it if it already exists.
    """

    s3 = boto3.client("s3", region_name=region)

    try:
        s3.head_bucket(Bucket=bucket_name)
        print("Bucket already exists:")
        print(bucket_name)

    except Exception:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={
                    "LocationConstraint": region
                }
            )

        print("Bucket created:")
        print(bucket_name)

    return bucket_name


def run_athena_query(query, database, output_location, region):
    """
    Run an Athena query and wait until it finishes.
    """

    athena = boto3.client("athena", region_name=region)

    response = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={
            "Catalog": "AwsDataCatalog",
            "Database": database
        },
        ResultConfiguration={
            "OutputLocation": output_location
        }
    )

    query_execution_id = response["QueryExecutionId"]

    while True:
        result = athena.get_query_execution(
            QueryExecutionId=query_execution_id
        )

        status = result["QueryExecution"]["Status"]["State"]

        if status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            break

        time.sleep(2)

    if status != "SUCCEEDED":
        raise RuntimeError(result["QueryExecution"]["Status"])

    return query_execution_id
