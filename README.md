# 🛒 E-Commerce Cart Abandonment Predictor

> 6-algorithm ML benchmark — individual model analysis — SHAP explainability — live predictions

**[🚀 Live Demo](https://ecommerce-cart-predictor-2rdexii9pkuaxgxtmkxdix.streamlit.app/)** &nbsp;·&nbsp; [LinkedIn](https://linkedin.com/in/gayathrikammidi) &nbsp;·&nbsp; [GitHub](https://github.com/gkammidi-prog)

---

## The Problem

E-commerce businesses lose billions annually to cart abandonment. Identifying which sessions are at risk before the customer leaves enables targeted interventions — discount coupons, reminders, personalized nudges. This system predicts abandonment risk in real time, benchmarks every major ML algorithm, and explains every single decision.

---

## Results

| Model | AUC-ROC | Recall | F1 Score | False Alarms |
|-------|---------|--------|----------|--------------|
| XGBoost | **0.999** | 0.988 | 0.703 | 134 |
| Gradient Boosting | 0.998 | **1.000** | 0.638 | 185 |
| Random Forest | 0.997 | 0.994 | 0.733 | 117 |
| Logistic Regression | 0.995 | 1.000 | 0.639 | 184 |
| Decision Tree | 0.986 | 0.982 | **0.748** | **105** |
| KNN | 0.848 | 0.681 | 0.301 | 463 |

> **Key insight:** Highest AUC does not always mean best model. Decision Tree had the best F1 and fewest false alarms despite lower AUC than XGBoost. The right model depends on the business objective — this benchmark documents every tradeoff so decisions are data-driven, not assumed.

---

## What This System Does

- Predicts **cart abandonment risk** from session-level behavior features
- Benchmarks **6 ML algorithms** on the same train/test split — fair, documented, reproducible
- Shows **individual model deep-dive** — ROC curve, Precision-Recall curve, Confusion Matrix, Feature Importance per model
- Explains every prediction using **SHAP waterfall charts** — audit-ready, not a black box
- Deployed as a **live 5-tab Streamlit dashboard** with real-time predictions

---

## Live Dashboard

```
Tab 1 — Individual Model Analysis
  Select any of 6 models → ROC curve · Confusion Matrix ·
  Precision-Recall curve · Feature Importance

Tab 2 — Head-to-Head Comparison
  All 6 models in one table · 4 metric bar charts · all ROC curves overlaid

Tab 3 — Why XGBoost Won
  Documented reasoning — why boosting beats linear models on this data,
  why KNN ranked last, what every tradeoff means for business decisions

Tab 4 — SHAP Explainability
  Global feature importance · SHAP distribution · per-session waterfall

Tab 5 — Live Predictor
  Input any session profile → get abandonment risk + SHAP explanation
```

👉 **[Try it live](https://ecommerce-cart-predictor-2rdexii9pkuaxgxtmkxdix.streamlit.app/)**

---

## Engineering Decisions

**6-model benchmark over single-model assumption**
Never assume one algorithm wins. Train all candidates on the same split, compare AUC-ROC, Precision, Recall, F1, and business metrics like false alarms. Let data decide — not convention.

**SMOTE over undersampling**
Cart abandonment is 1.7% of sessions — extreme class imbalance. Undersampling discards 98%+ of legitimate signal. SMOTE synthesises minority-class examples, preserving all available information while giving models enough abandoned sessions to learn real patterns.

**Recall as the primary metric**
On imbalanced data, accuracy is a misleading metric by design. A model predicting "will purchase" every time scores 98.3% accuracy and catches zero abandonment. Recall measures what actually matters — how many at-risk sessions are identified before the customer leaves.

**SHAP over built-in feature importance**
Built-in importance shows global averages across all predictions. SHAP produces per-session attribution — exactly which feature pushed this specific session's risk score up or down. Actionable for product, marketing, and engineering teams.

**Why KNN failed — the data preprocessing lesson**
KNN achieved AUC 0.848, missed 52 sessions, and fired 463 false alarms. Price features ($0–500) dominated distance calculations over event counts (0–50) without normalisation. This is not just an algorithm ranking — it is a data preprocessing insight that matters in every production ML system.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Modeling | XGBoost · scikit-learn · Logistic Regression · Decision Tree · Random Forest · Gradient Boosting · KNN |
| Imbalance | SMOTE (imbalanced-learn) |
| Explainability | SHAP — waterfall, summary, and distribution plots |
| Data | Pandas · NumPy |
| Visualization | Matplotlib · Seaborn |
| Deployment | Streamlit Cloud |

---

## Project Structure

```
ecommerce-cart-predictor/
├── streamlit_app.py     # Live dashboard — 5 tabs, all 6 models, SHAP
├── model.py             # Standalone training pipeline
├── requirements.txt     # Pinned dependencies
└── README.md
```

---

## Run Locally

```bash
git clone https://github.com/gkammidi-prog/ecommerce-cart-predictor
cd ecommerce-cart-predictor
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Place `2019-Nov.csv` from Kaggle into the `data/` folder for real data.
Without it, the app generates realistic synthetic data automatically.

---

## Dataset

| Dataset | Size | Source |
|---------|------|--------|
| eCommerce Behavior Data | 4M+ events · multi-category store | Kaggle (mkechinov) |

Features engineered at session level: event count, cart events, view events, average price, unique products viewed.

---

## Portfolio

| Project | Domain | Highlight |
|---------|--------|-----------|
| **E-Commerce Cart Predictor** *(this repo)* | Retail | 6-model benchmark · AUC 0.999 · Live |
| Credit Risk & Fraud Detection | Banking | AUC 0.869 · Fraud Recall 75% · Live |
| Medicare HCC Risk Score | Healthcare | Recall 90.5% · 71,518 encounters · Live |
| Hospital Readmission Predictor | Healthcare | SMOTE · XGBoost · Live |

---

## Author

**Gayathri Kammidi**
MS Computer Science · Governors State University · May 2026
4+ years in Data Engineering & ML — GCP · BigQuery · Airflow · Python · XGBoost

[LinkedIn](https://linkedin.com/in/gayathrikammidi) · [GitHub](https://github.com/gkammidi-prog)
