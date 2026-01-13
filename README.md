# Fraud Detection API

A production-ready **FastAPI service** for real-time fraud detection using an **XGBoost machine learning model**. The API accepts transaction-level features and returns a fraud probability score, a binary fraud decision, and a risk level classification.


## 🚀 Features

* Real-time fraud probability prediction
* Preprocessing and model logic bundled in a single pipeline
* Handles highly imbalanced fraud data
* Clear risk stratification (`Low`, `Medium`, `High`)
* Ready for integration into web, mobile, or payment systems


## 📥 API Input Features

| Feature Description|

| **TransactionAmt** | Transaction amount in the local currency. Larger or unusual amounts often correlate with fraud risk.                                       |
| **P_emaildomain**  | Domain of the purchaser’s email address (e.g., `gmail.com`, `yahoo.com`). Certain domains may show higher fraud patterns.                  |
| **C1**             | Count-based transactional feature representing customer activity frequency. Higher or abnormal values may indicate suspicious behavior.    |
| **C13**            | Aggregated behavioral feature derived from historical transaction patterns. Often useful for distinguishing normal vs fraudulent behavior. |
| **C14**            | Another historical aggregation feature capturing customer transaction characteristics over time.                                           |
| **card4**          | Card network type (e.g., `visa`, `mastercard`, `amex`). Some networks may have different fraud profiles.                                   |
| **card6**          | Card category (e.g., `debit`, `credit`). Credit and debit cards typically show different fraud risk patterns.                              |

> ⚠️ **Note:** Feature names and preprocessing must remain unchanged to ensure consistency between training and inference.


## 📤 API Response

```json
{
  "fraud_probability": 0.16904859244823456,
  "is_fraud": false,
  "risk_level": "Low"
}
```

* **fraud_probability** → Model-estimated likelihood of fraud
* **is_fraud** → Binary decision based on a configurable threshold (default: 0.5)
* **risk_level** → Human-readable risk band for downstream systems


## 📊 Model Performance (Test Set)

The model was evaluated with fraud-focused metrics suitable for highly imbalanced datasets:

* **Precision:** 0.1601
* **Recall:** 0.6891
* **F1-score:** 0.2599
* **ROC-AUC:** 0.8625

### Interpretation

* **High recall** ensures most fraudulent transactions are detected.
* **Moderate precision** is acceptable in fraud systems where recall is prioritized and false positives can be reviewed or filtered downstream.
* **Strong ROC-AUC** indicates excellent ranking ability for risk-based decisioning.


## 📈 Evaluation Plots Explained

### Confusion Matrix

Shows the count of:

* **True Positives:** Fraud correctly detected
* **False Positives:** Legitimate transactions flagged as fraud
* **False Negatives:** Missed fraud cases (most costly)
* **True Negatives:** Legitimate transactions correctly approved

Used to understand the trade-off between catching fraud and minimizing customer friction.


### ROC Curve

Plots **True Positive Rate vs False Positive Rate** across thresholds.

* A curve closer to the top-left indicates better performance.
* The ROC-AUC score summarizes the model’s ability to rank fraudulent transactions above legitimate ones.---

### Precision-Recall Curve

Focuses on performance for the **fraud class**:

* Shows how precision changes as recall increases.
* Especially important for imbalanced datasets like fraud detection.


### Feature Importance Plot

Displays the most influential features used by the XGBoost model.

* Helps explain which transaction attributes contribute most to fraud predictions.
* Useful for model interpretability, audits, and feature monitoring.


## ▶️ Running the API Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Swagger documentation:

```
http://127.0.0.1:8000/docs
```


## 🏁 Notes

* The model outputs **probability scores**, enabling flexible threshold tuning.
* Designed for extension with authentication, logging, monitoring, and model versioning.


**Author:** Abdulrahman Hayatu ML Engineer
**Status:** Production-ready