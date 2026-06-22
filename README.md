# Diabetes Readmission Prediction Using AWS SageMaker MLOps

An end-to-end Machine Learning Operations (MLOps) project built on AWS SageMaker to predict 30-day hospital readmissions for diabetic patients. This project demonstrates modern MLOps practices including data engineering, feature management, automated training pipelines, batch inference, monitoring, and CI/CD workflows.

---

## Team Members

| Name         | GitHub                         |
| ------------ | ------------------------------ |
| Divya Kamath | https://github.com/dkamath16   |
| Gaius Thomas | https://github.com/jeffiThomas |
| Ali Azizi    | https://github.com/al1az1z1    |

---

## Project Overview

Hospital readmissions are costly and often indicate that patients require additional care after discharge. The goal of this project is to predict whether a diabetic patient will be readmitted within 30 days using historical hospital encounter data.

Beyond predictive modeling, this project focuses on building a production-style machine learning system using AWS SageMaker and MLOps best practices.

### Business Objective

Identify high-risk diabetic patients before discharge so healthcare providers can:

* Schedule follow-up care
* Improve patient outcomes
* Reduce preventable readmissions
* Optimize healthcare resources

---

## Dataset

### Source

**UCI Machine Learning Repository**

Diabetes 130-US Hospitals Dataset

https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008

### Dataset Characteristics

* Over 100,000 patient encounters
* 130 hospitals across the United States
* Demographic information
* Admission information
* Laboratory results
* Medication information
* Readmission outcomes

### Target Variable

Original target:

```text
<30
>30
NO
```

Converted into binary classification:

```text
<30 = 1 (Readmitted within 30 days)
>30 = 0
NO  = 0
```

---

# Project Architecture

```text
Raw Dataset
      │
      ▼
 Amazon S3 Data Lake
      │
      ▼
 Data Engineering
 (Athena + EDA)
      │
      ▼
 Feature Engineering
      │
      ▼
 SageMaker Feature Store
      │
      ▼
 SageMaker Pipeline
 ├─ Preprocessing
 ├─ Training (XGBoost)
 ├─ Evaluation
 ├─ F1 Quality Gate
 ├─ Model Registration
 └─ Batch Transform
      │
      ▼
 Amazon S3 Predictions
      │
      ▼
 CloudWatch Monitoring
```

---

## Technologies Used

### AWS Services

* Amazon SageMaker
* SageMaker Pipelines
* SageMaker Feature Store
* Amazon S3
* Amazon Athena
* AWS Glue
* Amazon CloudWatch
* Batch Transform

### Machine Learning

* XGBoost
* Scikit-learn
* Pandas
* NumPy

### Development Tools

* Python
* Jupyter Notebooks
* GitHub
* Asana

---

## Data Engineering

The raw dataset is uploaded into Amazon S3 and organized as a data lake.

### Implemented Components

* Amazon S3 storage
* Athena database and external tables
* Exploratory Data Analysis (EDA)
* SQL-based analysis through Athena

Workflow:

```text
diabetic_data.csv
        │
        ▼
Amazon S3
        │
        ▼
Athena Queries
        │
        ▼
Exploratory Analysis
```

---

## Feature Engineering

The preprocessing workflow performs:

### Data Cleaning

* Removed unique identifiers:

  * patient_nbr
  * encounter_id

### Data Transformation

* Binary target conversion
* Missing value handling
* Boolean conversion
* Numeric feature alignment

### Encoding

One-hot encoding applied to categorical features such as:

* Race
* Gender
* Age
* Admission type
* Discharge disposition
* Medication information

---

## Feature Store

A SageMaker Feature Group was created to demonstrate centralized feature management.

### Purpose

* Feature governance
* Feature reuse
* Centralized feature schema management

### Note

The final production pipeline trains directly from Amazon S3 because the dataset is static and only a single model is deployed.

---

## Model Training

During exploratory modeling, multiple algorithms were evaluated.

### Baseline Model

* Logistic Regression

### Production Model

* XGBoost

XGBoost was selected because it provided stronger performance on structured healthcare data and was integrated into the automated SageMaker Pipeline.

---

## Data Splits

To reduce AWS costs while maintaining representative samples:

| Dataset             | Size   |
| ------------------- | ------ |
| Development Dataset | 15,000 |
| Production Dataset  | 6,000  |

Development split:

```text
Train       80%
Validation  10%
Test        10%
```

---

## Model Evaluation

The pipeline automatically generates an evaluation report containing:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

### Final XGBoost Results

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 0.748 |
| Precision | 0.187 |
| Recall    | 0.377 |
| F1 Score  | 0.250 |
| ROC-AUC   | 0.644 |

### Why Recall Matters

In healthcare, missing a high-risk patient can prevent timely intervention.

For this reason, Recall and F1 Score were treated as important business metrics.

---

## CI/CD Pipeline

The project implements an automated SageMaker Pipeline.

### Pipeline Steps

```text
Preprocess
    │
    ▼
Train XGBoost
    │
    ▼
Evaluate Model
    │
    ▼
F1 Threshold Check
    │
 ┌──┴───────────────┐
 │                  │
Pass             Fail
 │                  │
 ▼                  ▼
Batch Transform   Stop Pipeline
```

### Quality Gate

Models must achieve the required F1 threshold before continuing to deployment.

This demonstrates automated release control and CI/CD validation.

---

## Batch Inference

The project uses SageMaker Batch Transform rather than a continuously running endpoint.

### Why Batch Transform?

* Lower operational cost
* Better fit for healthcare reporting workflows
* Production-style scoring on unseen patient data

Outputs are written to Amazon S3 for downstream use.

---

## Monitoring

CloudWatch monitoring was implemented to track:

### Model Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

### Infrastructure Metrics

* Pipeline executions
* Processing jobs
* Training jobs
* Batch Transform jobs

A custom CloudWatch dashboard provides operational visibility into the ML workflow.

---
## Repository Structure
.
├── orchestrator.ipynb
│
├── notebooks/
│   ├── 0_Research_and_Prototyping.ipynb
│   └── 01_eda_diabetes_readmission.ipynb
│
├── src/
│   ├── __init__.py
│   ├── athena_setup.py
│   ├── config.py
│   ├── data_ingestion.py
│   ├── evaluation.py
│   ├── feature_store.py
│   ├── monitoring.py
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
│
├── screenshots/
│   ├── architecture.png
│   ├── Raw_data_location.png
│   ├── S3_Data_Lake.png
│   ├── athena-query-results.png
│   ├── diabetes-readmission-feature-group.png
│   ├── Successful-Pipeline-Execution.png
│   ├── Failed-Pipeline-Execution.png
│   ├── evaluation.png
│   ├── evaluation2.png
│   ├── Model-Artifact.png
│   ├── Batch-Predictions-Output.png
│   ├── CloudWatch-Metrics.png
│   └── SageMaker-Model.png
│
├── reports/
│   ├── AAI-540_ML_Design_Document.pdf
│   └── Diabetic_Readmission_Prediction_Final_Report.pdf
│
├── README.md
└── requirements.txt


### Project Workflow

| Component                           | Purpose                                                                                                                                                                   |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `0_Research_and_Prototyping.ipynb`  | Early experimentation, exploratory modeling, and comparison of candidate machine learning models.                                                                         |
| `01_eda_diabetes_readmission.ipynb` | Data ingestion, Amazon Athena integration, exploratory data analysis (EDA), and feature engineering.                                                                      |
| `src/`                              | Reusable Python modules for data preprocessing, feature engineering, training, evaluation, monitoring, and pipeline components.                                           |
| `orchestrator.ipynb`                | Main notebook that orchestrates the end-to-end SageMaker MLOps workflow including preprocessing, training, evaluation, CI/CD validation, batch inference, and monitoring. |
| `screenshots/`                      | Evidence of AWS resources, pipeline executions, Feature Store, CloudWatch monitoring dashboards, model artifacts, and batch inference outputs.                            |
| `reports/`                          | ML design documentation, project reports, and supporting project deliverables.                                                                                            |
|                                     |                                                                                                                                                                           |


## Future Enhancements

* Automated retraining with new hospital data
* Full SageMaker Model Monitor integration
* Data drift detection
* Prediction drift monitoring
* Bias and fairness analysis using SageMaker Clarify
* Real-time endpoint deployment
* Training directly from Feature Store
* Training on the full 100,000+ record dataset

---

## Disclaimer

This project was created for educational purposes as part of the AAI-540 Machine Learning Operations course.

The model is not intended for clinical decision-making and should not be used to make healthcare decisions without appropriate medical review.

---

## Course

**AAI-540 – Machine Learning Operations (MLOps)**
University of San Diego

---

## License

This repository is intended for educational and portfolio purposes.
