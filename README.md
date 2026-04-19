# MTM Risk Stratification System

> An interpretable clinical decision-support prototype that identifies patients at high risk for medication-related problems and prioritizes them for Medication Therapy Management (MTM) interventions.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## Project Overview

This project was developed as part of a Health Informatics practicum. It combines a validated rule-based clinical scoring engine with machine learning models to produce interpretable, actionable MTM risk classifications for simulated patient populations.

The system is designed to demonstrate how clinical decision support tools can be built transparently — prioritizing explainability and clinical validity over black-box complexity.

---

## What the System Does

- Ingests and cleans patient-level medication and clinical data
- Engineers clinically meaningful features (polypharmacy flags, adherence risk, comorbidity burden)
- Scores each patient using a validated weighted risk formula (Model C)
- Assigns risk tiers (Low / Medium / High) using percentile-based thresholds
- Applies safety rule overrides to catch clinically obvious high-risk cases
- Generates a plain-language explanation of each patient's top risk drivers
- Recommends a specific MTM intervention action per patient
- Trains and evaluates two ML models (Logistic Regression and Random Forest)
- Combines ML outputs into an ensemble probability score
- Flags patients where models disagree for human review
- Displays all results in an interactive multi-tab Streamlit dashboard

---

## Models Used

| Model | Type | ROC-AUC | Tier Agreement |
|---|---|---|---|
| Rule-based (Model C) | Weighted linear scoring | Baseline | Baseline |
| Logistic Regression | Supervised ML | 0.998 | 86% |
| Random Forest | Ensemble ML | 0.995 | 68% |
| Ensemble (LR + RF) | Averaged probability | — | 77% |

### Why Logistic Regression
Interpretable coefficients allow direct clinical explanation of feature influence. Widely used in validated clinical risk scores (CHADS2, CHA2DS2-VASc). Appropriate complexity for 4 features and 800 patients.

### Why Random Forest
Detects non-linear feature interactions that logistic regression misses. Provides independent feature importance scores that can be compared against LR coefficients. More robust to edge cases.

### Why Ensemble
Averages LR and RF probability outputs to produce a more balanced estimate. Reduces the impact of individual model biases. Patients where all models agree are flagged as high-confidence classifications.

---

## Feature Importance

Both models independently agreed on feature ranking:

| Feature | LR Coefficient | RF Importance | Clinical Meaning |
|---|---|---|---|
| last_hospital | +4.422 | 0.418 | Recent hospitalization is the strongest risk signal |
| adherence_score | −3.017 | 0.272 | Higher adherence strongly reduces risk |
| med_count | +2.697 | 0.194 | More medications increases risk |
| comorbidity_count | +1.761 | 0.116 | More chronic conditions increases risk |

---

## Dashboard

The Streamlit dashboard has three tabs:

**Overview & Clinical Depth**
- Summary metrics including all-model agreement rate
- Risk tier distribution, age distribution, clinical factor averages
- Adherence vs risk score scatter plot
- Hospitalization rate by tier
- High-priority patient table

**ML Insights**
- Model performance comparison
- Probability distribution histograms for all three models
- LR vs RF probability scatter plot
- Feature importance comparison table
- Disagreement case deep-dive table

**Patient Detail**
- Color-coded risk badge and agreement indicator
- Four gauges: rule-based score, LR probability, RF probability, ensemble probability
- Patient vs population average chart
- Risk explanation and MTM recommendation
- Full model comparison with consensus alert

---

## Project Structure

```
MTM_Risk_project/
├── app.py                  # Streamlit dashboard
├── main_pipeline.py        # Data pipeline, scoring, and ML training (run in Google Colab)
├── generate_mock_data.py   # Synthetic data generator (800 patients)
├── requirements.txt        # Python dependencies
├── data/
│   ├── raw.csv             # Raw synthetic patient data
│   └── scored_data.csv     # Fully scored and ML-enhanced output
└── README.md
```

---

## How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/mtm-risk-stratification.git
cd mtm-risk-stratification
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the pipeline** (in Google Colab or locally)

Open `main_pipeline.py` and run all cells. This generates `data/scored_data.csv`.

**4. Launch the dashboard**
```bash
streamlit run app.py
```

---

## Data

All data used in this project is **fully synthetic**, generated programmatically using a custom mock data generator. No real patient data was used at any stage. The generator produces clinically realistic distributions with intentional noise to simulate real-world edge cases.

**Dataset:** 800 simulated patients  
**Features:** age, medication count, adherence score, comorbidity count, recent hospitalization, HbA1c  
**Noise:** 25% of patients assigned intentional clinical edge-case patterns to create realistic model disagreement

---

## Important Limitations

- **Synthetic data only.** Real-world clinical performance cannot be assumed.
- **Proxy labels.** ML models are trained on expert-defined rule-based labels, not independent clinical outcomes. This is a recognized approach in clinical AI prototyping and is acknowledged transparently throughout the project.
- **Not for clinical use.** This is a practicum prototype for educational and portfolio purposes only.

---

## Tech Stack

- **Language:** Python 3
- **Dashboard:** Streamlit
- **ML:** scikit-learn (Logistic Regression, Random Forest)
- **Visualization:** Plotly
- **Data:** pandas, numpy
- **Development:** Google Colab (pipeline), local (dashboard)

---

## Update
MTM_app.py — Self-Contained Risk Stratification Dashboard
MTM_app.py is the final version of the MTM Risk Stratification System — a fully self-contained Streamlit dashboard that runs the complete clinical AI pipeline from a single file upload.
How it works
Upload a raw patient CSV and the app automatically runs the full pipeline — data cleaning, feature engineering, rule-based scoring, safety rule overrides, explainability, MTM recommendations, and three machine learning models — then displays the results as an interactive dashboard. No external scripts, no pre-generated files, no setup required.
What it does

Accepts any patient CSV with the required clinical columns
Runs the complete MTM risk stratification pipeline internally
Scores patients as Low, Medium, or High risk using Model C (adherence-focused weighted scoring)
Applies three safety rule overrides to catch under-classified high-risk cases
Trains Logistic Regression and Random Forest models with 5-fold cross-validation
Combines models into an ensemble probability score with per-patient confidence flagging
Displays results across three dashboard tabs — population overview, ML insights, and per-patient detail
Provides a one-click download of the fully scored results CSV

---

## Author

Master's Student — Health Informatics (Pharmacy background)  
Health Informatics Practicum · April 2026

---

## Acknowledgements

This project was developed as a practicum proof-of-concept to functional prototype. The clinical logic is informed by published MTM literature, CMS Star Ratings adherence measures, and standard pharmacy practice guidelines.
