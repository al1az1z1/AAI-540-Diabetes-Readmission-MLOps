
# ============================================================
# Project Configuration
# ============================================================

RAW_FILE = "diabetic_data.csv"

TARGET = "readmitted"

RANDOM_STATE = 42

SAMPLE_SIZE = 15000

PRODUCTION_SIZE = 6000

PIPELINE_NAME = "DiabetesReadmissionPipeline"

MODEL_PACKAGE_GROUP_NAME = "DiabetesReadmissionModelPackageGroup"

FEATURE_GROUP_NAME = "diabetes-readmission-feature-group"

ATHENA_DATABASE = "aai540_diabetes_db"

ATHENA_TABLE = "diabetes_raw"

CLOUDWATCH_NAMESPACE = "AAI540/DiabetesReadmissionPipeline"

DASHBOARD_NAME = "AAI540-DiabetesReadmission-Pipeline-Monitoring"

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
