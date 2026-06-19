
import pandas as pd

from src.utils import run_athena_query


def create_athena_database_and_table(
    bucket_name,
    region,
    database_name,
    table_name
):
    """
    Create Athena database and external table
    for the raw diabetes CSV stored in S3.
    """

    athena_output = f"s3://{bucket_name}/athena-query-results/"

    create_db_query = f"""
    CREATE DATABASE IF NOT EXISTS {database_name}
    """

    run_athena_query(
        query=create_db_query,
        database="default",
        output_location=athena_output,
        region=region
    )

    print("Athena database ready:")
    print(database_name)

    create_table_query = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {database_name}.{table_name} (
        encounter_id BIGINT,
        patient_nbr BIGINT,
        race STRING,
        gender STRING,
        age STRING,
        weight STRING,
        admission_type_id INT,
        discharge_disposition_id INT,
        admission_source_id INT,
        time_in_hospital INT,
        payer_code STRING,
        medical_specialty STRING,
        num_lab_procedures INT,
        num_procedures INT,
        num_medications INT,
        number_outpatient INT,
        number_emergency INT,
        number_inpatient INT,
        diag_1 STRING,
        diag_2 STRING,
        diag_3 STRING,
        number_diagnoses INT,
        max_glu_serum STRING,
        A1Cresult STRING,
        metformin STRING,
        repaglinide STRING,
        nateglinide STRING,
        chlorpropamide STRING,
        glimepiride STRING,
        acetohexamide STRING,
        glipizide STRING,
        glyburide STRING,
        tolbutamide STRING,
        pioglitazone STRING,
        rosiglitazone STRING,
        acarbose STRING,
        miglitol STRING,
        troglitazone STRING,
        tolazamide STRING,
        examide STRING,
        citoglipton STRING,
        insulin STRING,
        glyburide_metformin STRING,
        glipizide_metformin STRING,
        glimepiride_pioglitazone STRING,
        metformin_rosiglitazone STRING,
        metformin_pioglitazone STRING,
        change STRING,
        diabetesMed STRING,
        readmitted STRING
    )
    ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
    WITH SERDEPROPERTIES (
        "separatorChar" = ",",
        "quoteChar" = "\\""
    )
    LOCATION 's3://{bucket_name}/raw/diabetes/'
    TBLPROPERTIES (
        "skip.header.line.count" = "1"
    )
    """

    run_athena_query(
        query=create_table_query,
        database=database_name,
        output_location=athena_output,
        region=region
    )

    print("Athena table ready:")
    print(f"{database_name}.{table_name}")

    return athena_output


def run_athena_eda_query(
    bucket_name,
    region,
    database_name,
    table_name
):
    """
    Run a lightweight Athena EDA query and return
    the S3 path of the query result.
    """

    athena_output = f"s3://{bucket_name}/athena-query-results/"

    eda_query = f"""
    SELECT
        readmitted,
        COUNT(*) AS record_count,
        ROUND(AVG(time_in_hospital), 2) AS avg_time_in_hospital,
        ROUND(AVG(num_medications), 2) AS avg_num_medications,
        ROUND(AVG(number_inpatient), 2) AS avg_number_inpatient
    FROM {database_name}.{table_name}
    GROUP BY readmitted
    ORDER BY record_count DESC
    """

    query_id = run_athena_query(
        query=eda_query,
        database=database_name,
        output_location=athena_output,
        region=region
    )

    result_s3_uri = f"{athena_output}{query_id}.csv"

    print("Athena EDA query complete.")
    print("Query result stored at:")
    print(result_s3_uri)

    return result_s3_uri
