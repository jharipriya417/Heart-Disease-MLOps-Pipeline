# Explainable and Drift-Aware MLOps Pipeline for Heart Disease Prediction

An end-to-end MLOps prototype that predicts heart disease from patient health parameters, explains each prediction, and monitors incoming data for drift once deployed on AWS. It uses an Explainable Boosting Machine (EBM) as the main model, with Logistic Regression as a baseline.

## Highlights
- **Explainable model:** EBM with global feature importance and per-patient feature contributions
- **Experiment tracking:** MLflow records parameters, metrics and artifacts
- **Cloud deployment:** FastAPI service on Amazon EC2; model stored in Amazon S3 (SKOPS format)
- **Web interface:** HTML, CSS and JavaScript form to enter patient details and view predictions with charts
- **Drift monitoring:** Evidently AI compares production inputs with reference data
- **Logging and alerts:** Amazon CloudWatch collects application logs; Amazon SNS sends an email when drift is detected

## Architecture
1. Dataset is loaded from Amazon S3, preprocessed (cleaning, feature engineering, train-test split, scaling and encoding)
2. EBM and Logistic Regression are trained and evaluated; metrics are tracked in MLflow
3. The selected EBM model is serialized with SKOPS and uploaded to S3
4. The FastAPI app on EC2 loads the model and serves `/predict`, `/explain/global` and drift monitoring endpoints
5. Every prediction (inputs, output, probability, timestamp) is logged to `prediction_logs.csv`
6. Evidently AI checks drift once at least 30 production records exist:

| Condition | Status |
|---|---|
| Fewer than 30 records | `INSUFFICIENT_DATA` |
| Drifted features below 20% | `NORMAL` |
| Drifted features at or above 20% | `DRIFT_ALERT` (SNS email sent) |

## Results

| Metric | EBM (primary) | Logistic Regression (baseline) |
|---|---|---|
| Accuracy | 87.04% | 85.19% |
| Precision | 81.48% | 78.57% |
| Recall | 91.67% | 91.67% |
| F1-score | 86.27% | 84.62% |
| ROC-AUC | 90.83% | 89.86% |

- Mean inference latency: about 1.0 ms for EBM
- Drift check on 30 production records: 0 of 13 monitored features drifted (0.00%), so status `NORMAL`

## Tech Stack
Python, scikit-learn, InterpretML (EBM), MLflow, FastAPI, Evidently AI, AWS (EC2, S3, CloudWatch, SNS), HTML, CSS, JavaScript

## How to Run
```bash
git clone https://github.com/jharipriya417/Heart-Disease-MLOps-Pipeline.git
cd Heart-Disease-MLOps-Pipeline
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn main:app --reload    # change main:app to your actual FastAPI entry point
```
Open `http://127.0.0.1:8000/docs` for the API documentation. AWS credentials and the S3 bucket name must be set as environment variables. Never commit keys to the repository.

## Limitations
This is a foundational MLOps prototype. It detects input drift and sends alerts, but it does not retrain or redeploy the model automatically. Detected drift does not by itself mean the model's accuracy has dropped. The model is for learning and demonstration only and must not be used for medical decisions.
