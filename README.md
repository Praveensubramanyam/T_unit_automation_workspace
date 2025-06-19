
# 🚀 Tide Dynamic Pricing Optimization System

[![Azure ML](https://img.shields.io/badge/AzureML-Deployed-blue?logo=microsoftazure)](https://azure.microsoft.com/)
[![MLflow Tracking](https://img.shields.io/badge/MLflow-Tracking-success?logo=mlflow)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📌 Project Overview

This project implements a dynamic pricing engine for Tide detergent, powered by machine learning. It leverages behavioral analytics, inventory status, competitor pricing, and customer demand to optimize price recommendations. The solution includes real-time APIs, automated pipelines, retraining logic, and full observability via Azure ML and MLflow.

---

## 🧱 Project Structure

```
.
├── api/                      # FastAPI backend for serving ML models
├── conda/                   # Conda environment definition for Azure ML
├── data/                    # Data ingestion scripts
├── frontend/                # Streamlit UI for pricing visualization
├── jobs/                    # YAML-based Azure ML job definitions
├── mlruns/                  # MLflow experiment tracking logs
├── notebooks/               # Jupyter notebooks for data exploration
├── src/                     # Core codebase
│   ├── config/              # App and environment settings
│   ├── data/                # Preprocessing logic
│   ├── evaluation/          # Custom evaluation metrics
│   ├── features/            # Feature engineering & time series
│   ├── models/              # Model training & MLflow tracking
│   ├── pipeline/            # Orchestration & ML pipeline script
│   ├── utils/               # Validators, setup utilities, helpers
├── tests/                   # Unit and integration tests
├── workflows/               # CI/CD GitHub Actions pipeline
├── project_setup.py         # Project initialization script
├── azure_ml_deployment.py   # Managed endpoint deployment logic
├── automated_retraining.py  # Retraining pipeline with triggers
├── monitoring_system.py     # App Insights + model drift monitoring
├── README.md
```

---

## 🔧 Setup Instructions

### 1. Clone & Install

```bash
git clone https://github.com/your-org/dynamic-pricing-tide.git
cd dynamic-pricing-tide
pip install -r requirements.txt
```

### 2. Initialize Project

```bash
python src/project_setup.py
```

Sets up:
- Folder structure
- Azure Key Vault access
- MLflow experiment logging
- JSON-structured logging via Azure App Insights

---

## ⚙️ Running the Application

### 🧠 Train Model

```bash
python src/main.py
```

### 🧪 Serve via API

```bash
cd api
uvicorn main:app --reload
```

### 🎛️ Run Streamlit UI

```bash
cd frontend
streamlit run streamlit_app.py
```

---

## 📈 ML Workflow Highlights

- **MLflow Tracking**: Experiments auto-logged with parameters, metrics, and artifacts.
- **Model Tuning**: GridSearchCV over multiple models: RandomForest, XGBoost, Ridge, Lasso, SVR.
- **Feature Engineering**:
  - Price Elasticity Proxy
  - Customer Journey Signals
  - Inventory Analytics (FillRate, Backorders, Stockouts)
  - Time Series Decomposition (via STL)

---

## 🧪 Testing & Quality

```bash
pytest tests/
```

Tests include:
- Unit tests for preprocessing and API logic
- Integration test for model prediction pipeline
- Smoke test for endpoint health

---

## 📦 Deployment Pipeline

- **Azure ML** for model packaging and deployment
- **Blue-Green Deployment** strategy using managed endpoints
- **Rollback logic** based on App Insights and MLflow performance comparison

Run deployment:

```bash
python azure_ml_deployment.py
```

---

## 🔄 Retraining Logic

- Scheduled + Drift + Performance Degradation triggers
- Champion-Challenger with A/B testing
- MLflow-driven model promotion and rollback
- Business impact checks before promotion

Trigger:

```bash
python automated_retraining.py
```

---

## 📊 Monitoring (via Azure App Insights)

Tracked:
- Revenue impact
- Demand forecast error
- Pricing accuracy
- Customer churn proxy
- Latency, error rates, throughput
- Alerting severity levels (warnings, critical)

Run:

```bash
python monitoring_system.py
```

---

## 🧬 API Reference

**POST** `/predict`

```json
{
  "Date": "2025-06-14",
  "SellingPrice": 58.5,
  "Brand": "Tide",
  "Demand": 120
}
```

Returns:

```json
{
  "PredictedUnitsSold": 135
}
```

---

## 📖 Documentation Coverage

- ✅ Architecture diagram (see above)
- ✅ Setup, configuration, and `.env` usage
- ✅ User manual: run training, APIs, UI
- ✅ Dev manual: folder structure, test strategy, CI/CD
- ✅ Monitoring and retraining ops
- ✅ ML model lifecycle flow
- ✅ Licensing and contribution guide

---

## 👩‍💻 Contributing

1. Fork the repo
2. Create a feature branch
3. Commit + test + lint
4. Open a PR with clear title + description

---

## 📄 License

Licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🤖 Copilot Assistance

This project was developed with GitHub Copilot assistance for:

- Folder structure generation
- Boilerplate scaffolding
- ML pipeline construction
- Logging and monitoring integrations
- Documentation generation

---
