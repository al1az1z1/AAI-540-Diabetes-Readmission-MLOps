
import os
import boto3


def upload_raw_dataset(raw_file, bucket_name, region):
    """
    Upload the raw diabetes dataset to the S3 data lake.
    """

    if not os.path.exists(raw_file):
        raise FileNotFoundError(
            f"Raw file not found: {raw_file}. "
            "Make sure diabetic_data.csv is in the project root."
        )

    s3 = boto3.client("s3", region_name=region)

    raw_s3_key = "raw/diabetes/diabetic_data.csv"

    s3.upload_file(
        raw_file,
        bucket_name,
        raw_s3_key
    )

    raw_s3_uri = f"s3://{bucket_name}/{raw_s3_key}"

    print("Raw dataset uploaded to S3:")
    print(raw_s3_uri)

    return raw_s3_uri
