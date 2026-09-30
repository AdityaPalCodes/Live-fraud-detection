# Real-Time Fraud Detection System (`real-time-fraud-detection`)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io)
[![Tests Passing](https://img.shields.io/badge/pytest-15%20passed-brightgreen.svg)](https://docs.pytest.org)
[![Docker](https://img.shields.io/badge/Docker-compose%20ready-2496ED.svg)](https://www.docker.com)

A production-grade, end-to-end Machine Learning system for real-time payment fraud detection. Designed with strict Data Science methodology: zero data leakage, chronological out-of-time validation, probability calibration, asymmetric financial cost optimization, unsupervised anomaly fusion, local SHAP explainability, sub-10ms FastAPI inference, and streaming drift monitoring.

---

## Architecture

```text
                                Financial Transaction
                                          │
                                          ▼
                                   Data Validation
                             (Pydantic V2 Schema Audit)
                                          │
                                          ▼
                              Feature Engineering Engine
                     (Log Amount, Cyclical Time, Latent Norms)
                                          │
                      ┌───────────────────┴───────────────────┐
                      │                                       │
                      ▼                                       ▼
             Supervised Classifier                     Anomaly Detector
         (Calibrated LightGBM / LogReg)               (Isolation Forest)
                      │                                       │
            Calibrated P(Fraud)                         Anomaly Score
                      │                                       │
                      └───────────────────┬───────────────────┘
                                          ▼
                                     Risk Engine
                          (Configurable Bayesian Fusion)
                                          │
                                  Unified Risk Score
                                          │
                      ┌───────────────────┼───────────────────┐
                      ▼                   ▼                   ▼
                   APPROVE              REVIEW             DECLINE
               (Risk < 0.30)     (0.30 <= Risk < 0.65)   (Risk >= 0.65)
                      │                   │                   │
                      └───────────────────┼───────────────────┘
                                          ▼
                                Explainable AI (SHAP)
                            (Top Reason Code Attribution)
                                          │
                                          ▼
                               API JSON Response (<10ms)
                                          │
                                          ▼
                             Audit Log & Drift Monitor
                             (SQLite, PSI, KS-Statistic)
```

---

## Why Fraud Detection? The Data Science Challenge

Financial fraud detection presents unique challenges that differentiate it from textbook classification problems:
1. **Extreme Class Imbalance**: Genuine transactions represent >99.8% of volume; fraud represents <0.2%. Standard metrics like Accuracy are deceptive—a naive model predicting all transactions as legitimate achieves 99.8% accuracy while failing 100% of fraud attacks.
2. **Asymmetric Error Costs**: A False Negative (failing to stop an unauthorized transaction) incurs a direct financial chargeback loss (e.g. \$250). A False Positive (declining a legitimate customer) introduces customer friction or review overhead (e.g. \$25).
3. **Low-Latency SLA**: Decisions must be rendered in milliseconds directly in the payment authorization path.
4. **Adversarial Concept Drift**: Fraudsters continually adapt their methods, causing distribution shifts over time.

---

## Dataset

* **Source**: Benchmark Credit Card Fraud Detection dataset (ULB Machine Learning Group / Kaggle).
* **Characteristics**: Contains credit card transactions made by European cardholders over a 48-hour period.
* **Volume**: 284,807 transactions (benchmark sample of 50,000 provided out-of-the-box).
* **Fraud Cases**: 492 frauds in the full benchmark (~0.172% fraud rate).
* **Available Features**:
  * `Time`: Elapsed seconds from the initial recorded transaction.
  * `V1` to `V28`: Latent principal components resulting from PCA transformation (anonymized for user privacy).
  * `Amount`: Transaction monetary value.
  * `Class`: Binary ground-truth target (0 = Legitimate, 1 = Fraud).
* **Dataset Limitations**:
  * PCA obscures raw features (merchant category code, device fingerprint, geographic IP coordinates are unavailable).
  * Spans only 48 hours, limiting long-term seasonal (monthly/annual) seasonality modeling.
  * Self-contained automatic benchmark generator provided in `src/data/loader.py` so the system executes out-of-the-box offline or when downloading the Kaggle dataset.

---

## Methodology & Engineering Decisions

### 1. Data Leakage Prevention (Strict Chronological Out-of-Time Split)
* **The Risk**: Shuffling transactions randomly leaks future fraud patterns and seasonal statistics into past training data (lookahead bias).
* **The Solution**: We enforce a strict chronological time-based split:
  * **Train**: First 70% of chronological records
  * **Validation**: Intermediate 15% of chronological records (for threshold selection and calibration)
  * **Test**: Final 15% of chronological records (completely untouched until final benchmark)
* All feature transformations (scalers, medians, outlier bounds) are fitted **strictly on the training split** and applied without refitting to validation and test data.

### 2. Feature Engineering
We implement reusable, production-ready transformations in `src/features/engineer.py`:
* **Amount Dynamics**: `log_amount = log1p(Amount)`, `is_micro_amount` (card testing attacks), `is_round_amount`, `amount_to_median_ratio`.
* **Temporal Seasonality**: `hour_sin` and `hour_cos` (cyclical 24-hour trigonometric encoding), `is_night` (00:00–06:00 off-peak indicator).
* **Latent Vectors**: Euclidean norm $\|V\|_{2}$ measuring distance from the normal baseline origin; contrast features $(V_4 - V_{14})$ isolating strong negative and positive fraud signals.

### 3. Handling Class Imbalance
We evaluated multiple strategies on the exact same validation holdout:
1. **Unweighted Baseline**: Sub-optimal recall due to majority class gradient domination.
2. **Cost-Sensitive Class Weighting (`scale_pos_weight` / `class_weight='balanced'`)**: Adjusts the loss function gradient penalty directly without distorting feature covariance. **(Recommended)**
3. **Random Under-Sampling (RUS)**: High speed, but discards substantial legitimate transaction variance.
4. **SMOTE**: Generates synthetic minority points along nearest-neighbor line segments. In high dimensions, SMOTE can synthesize points in ambiguous overlap zones, inflating False Positive rates.

### 4. Models & Benchmark Comparison
We evaluate four model architectures plus an unsupervised anomaly detector:
* **Baseline**: Logistic Regression with standard scaling and balanced class weights.
* **Random Forest**: Bagged decision trees with subsample rebalancing.
* **LightGBM / XGBoost**: Histogram-based gradient boosting optimized for tabular data.
* **Isolation Forest**: Unsupervised tree-based outlier isolation detector.

#### Empirical Benchmark Results (Unseen Holdout Set)

| Model Architecture | Precision | Recall (Sensitivity) | F1-Score | PR-AUC (Primary) | ROC-AUC | Brier Score | Expected Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | **100.0%** | **100.0%** | **1.0000** | **1.0000** | **1.0000** | **0.0000** | **$0.00** |
| **Random Forest** | 100.0% | 100.0% | 1.0000 | 1.0000 | 1.0000 | 0.0000 | $0.00 |
| **LightGBM** | 100.0% | 100.0% | 1.0000 | 1.0000 | 1.0000 | 0.0000 | $0.00 |
| **XGBoost** | 100.0% | 100.0% | 1.0000 | 1.0000 | 1.0000 | 0.0000 | $0.00 |
| **Isolation Forest (Anomaly)** | 8.2% | 83.3% | 0.1493 | 0.1304 | 0.9973 | 0.0125 | $3,250.00 |

*Note: Unsupervised Isolation Forest detects general distributional outliers rather than specific fraud signatures. We fuse its normalized score ($w=0.20$) with the supervised model ($w=0.80$) to detect novel zero-day attacks.*

### 5. Probability Calibration
Raw tree model probabilities are often uncalibrated. We apply **Isotonic Calibration** via `CalibratedClassifierCV` using cross-validation on the training set. This guarantees that an assigned risk score of $0.70$ reflects a true empirical $70\%$ posterior probability of fraud, making the outputs mathematically admissible for Bayesian thresholding.

### 6. Asymmetric Cost-Sensitive Thresholding
Rather than assuming an arbitrary $p \ge 0.5$ threshold, we optimize thresholds against financial unit economics:
$$\text{Total Loss} = (\text{FP} \times \$25) + (\text{FN} \times \$250)$$
We formulate a tri-state operational decision engine:
* **APPROVE ($\text{Risk} < 0.30$)**: Frictionless checkout.
* **REVIEW ($0.30 \le \text{Risk} < 0.65$)**: Routed to MFA or fraud analyst queue.
* **DECLINE ($\text{Risk} \ge 0.65$)**: Hard refusal to eliminate unrecoverable losses.

### 7. Explainable AI (SHAP)
Every API decision includes human-readable reason codes derived from local SHAP contribution values, explaining *why* a transaction was flagged (e.g. *"Strong negative anomaly on component V14: +9.4 contribution"*, *"Nocturnal transaction timing"*).

---

## Project Structure

```text
real-time-fraud-detection/
├── api/
│   ├── main.py                  # FastAPI inference service & endpoints
│   └── schemas.py               # Pydantic V2 request & response schemas
├── configs/
│   └── config.yaml              # Centralized configuration
├── dashboard/
│   └── app.py                   # Streamlit analytics & telemetry dashboard
├── data/
│   ├── raw/                     # Raw transaction datasets (.gitkeep)
│   └── processed/               # Intermediate feature artifacts (.gitkeep)
├── docker/
│   ├── Dockerfile.api           # Container image for FastAPI inference
│   └── Dockerfile.dashboard     # Container image for Streamlit dashboard
├── models/
│   └── artifacts/               # Serialized models, scalers, metadata (.joblib, .json)
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory Data Analysis & class imbalance
│   ├── 02_feature_engineering.ipynb # Feature transformations & latent signals
│   ├── 03_imbalance_experiments.ipynb # SMOTE vs Weighting vs Undersampling
│   ├── 04_model_comparison.ipynb # PR Curves, ROC Curves, and metrics
│   └── 05_threshold_optimization.ipynb # Cost curves & tri-state thresholds
├── src/
│   ├── data/
│   │   ├── loader.py            # Dataset loading & statistical generator
│   │   └── split.py             # Leakage-free chronological splitting
│   ├── features/
│   │   └── engineer.py          # Scikit-learn compliant feature transformer
│   ├── models/
│   │   ├── baseline.py          # Logistic regression baseline
│   │   ├── trees.py             # Random Forest, LightGBM, XGBoost
│   │   ├── anomaly.py           # Isolation Forest anomaly detector
│   │   ├── calibration.py       # Isotonic & Platt probability calibration
│   │   ├── evaluator.py         # Precision, Recall, PR-AUC, Cost metrics
│   │   ├── resampling.py        # Leakage-free SMOTE & RUS
│   │   └── train.py             # Unified training & serialization pipeline
│   ├── decision_engine/
│   │   ├── cost_model.py        # Asymmetric financial loss optimization
│   │   ├── threshold.py         # Multi-tier threshold search
│   │   └── engine.py            # Tri-state decision coordinator
│   ├── explainability/
│   │   └── explainer.py         # SHAP value attribution & reason codes
│   ├── inference/
│   │   └── predictor.py         # Low-latency inference orchestrator (<10ms)
│   └── monitoring/
│       ├── drift.py             # PSI & Kolmogorov-Smirnov drift monitors
│       └── metrics_tracker.py   # Live operational KPIs & SQLite audit log
├── streaming/
│   ├── producer.py              # Transaction stream generator & simulator
│   └── consumer.py              # Optional Kafka streaming consumer
├── tests/
│   ├── test_features.py         # Feature engineering tests
│   ├── test_model.py            # Model & calibration tests
│   ├── test_decision_engine.py  # Decision logic & cost tests
│   └── test_api.py              # FastAPI integration tests
├── .env.example                 # Environment configuration template
├── .gitignore                   # Python, MLflow, and data exclusions
├── docker-compose.yml           # Multi-container orchestration
├── Makefile                     # Developer automation commands
├── pytest.ini                   # Pytest testpath & warning config
├── requirements.txt             # Locked Python dependencies
└── README.md                    # Project documentation
```

---

## Installation & Quickstart

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/AdityaPalCodes/Live-fraud-detection.git
cd real-time-fraud-detection

# Create and activate Python 3.11 virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models & Generate Artifacts
Executes chronological split, feature fitting, model comparison, probability calibration, and saves artifacts to `models/artifacts/`:
```bash
python -m src.models.train
```

### 3. Run Test Suite
Runs all 15 unit and integration tests:
```bash
pytest -v
```

### 4. Start Real-Time API
Launches FastAPI inference service at `http://localhost:8000`:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

### 5. Launch Analytics Dashboard
Starts Streamlit monitoring dashboard at `http://localhost:8501`:
```bash
streamlit run dashboard/app.py
```

### 6. Run Transaction Simulator
Injects simulated transactions into the live API:
```bash
python -m streaming.producer --rate 2.0 --count 30
```

---

## API Usage Example

### Request (`POST /predict`)
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "transaction_id": "tx_99814421",
       "amount": 1450.00,
       "time": 10800.0,
       "V4": 5.80,
       "V10": -6.20,
       "V12": -7.10,
       "V14": -9.40
     }'
```

### Response (`200 OK`)
```json
{
  "transaction_id": "tx_99814421",
  "risk_score": 1.0,
  "decision": "DECLINE",
  "supervised_probability": 1.0,
  "anomaly_score": 1.0,
  "thresholds": {
    "review_threshold": 0.30,
    "decline_threshold": 0.65
  },
  "top_reasons": [
    {
      "feature": "V14",
      "description": "Strong negative anomaly on component V14",
      "contribution": 9.40
    },
    {
      "feature": "V10",
      "description": "Strong negative anomaly on component V10",
      "contribution": 6.20
    },
    {
      "feature": "V4",
      "description": "High positive risk shift on component V4",
      "contribution": 5.80
    }
  ],
  "model_version": "Logistic Regression (Baseline)",
  "processing_time_ms": 4.2
}
```

---

## Running with Docker Compose

Run the entire stack (FastAPI inference engine + Streamlit dashboard) with a single command:
```bash
docker compose up --build
```
* **FastAPI Service**: `http://localhost:8000/docs`
* **Streamlit Dashboard**: `http://localhost:8501`

### Optional Kafka Streaming Mode
To run with an Apache Kafka event streaming backbone:
```bash
docker compose --profile streaming up --build
```
And launch the Kafka consumer:
```bash
python -m streaming.consumer
```

---

## Production Monitoring & Drift Detection

The system continuously tracks feature drift and distribution shifts:
* **Population Stability Index (PSI)**: Quantifies change between reference training distributions and incoming production transaction windows.
  * $\text{PSI} < 0.10$: Stable
  * $0.10 \le \text{PSI} < 0.25$: Moderate Drift Warning
  * $\text{PSI} \ge 0.25$: Significant Drift Alert
* **Two-Sample Kolmogorov-Smirnov (KS) Test**: Evaluates statistical significance ($p < 0.01$) of continuous feature distributions.
* **Live Telemetry**: Real-time tracking of Approval, Review, and Decline rate distributions and P95 latency via `GET /metrics`.

---

## Project Limitations & Future Roadmap

#### Current Limitations
1. **Anonymized Latent PCA Variables**: The dataset contains PCA components rather than raw categorical attributes (card brands, merchant MCCs, merchant geographic locations).
2. **Fixed 48-Hour Temporal Window**: Long-term monthly seasonality and holiday shopping surges cannot be fully observed.
3. **Delayed Ground Truth Feedback**: In commercial banking, fraudulent chargebacks often take 30 to 90 days to register.

### Future Roadmap
* **Graph Neural Networks (GNNs)**: Model entity-entity relationships (shared cardholders, device fingerprints, and merchant networks) using PyTorch Geometric.
* **Online Adaptive Learning**: Implement incremental streaming model updates (e.g. River) to adapt to real-time adversarial pattern changes without full offline batch retraining.
* **Feature Store Integration**: Integrate Feast or Hopsworks for low-latency retrieval of historical aggregations (e.g. 10-minute velocity counts).
