# FintechSurvive AI - Nigerian Fintech Startup Survival Predictor

A data science web application that predicts whether a Nigerian fintech startup will **Survive or Fail** using XGBoost and SHAP explainability.

Built with Python and Streamlit, deployed on Streamlit Community Cloud.

---

## Live Demo

[Click here to view the app](https://share.streamlit.io)

---

## Overview

This project builds an end-to-end machine learning classification pipeline that predicts the survival outcome of a Nigerian fintech startup based on 8 key signals including funding, team size, market segment, CBN licensing, and revenue status.

It uses SHAP (SHapley Additive exPlanations) to explain exactly which factors are pushing the prediction towards survival or failure.

---

## Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | XGBoost Classifier |
| Estimators | 200 trees |
| Max Depth | 4 |
| Learning Rate | 0.1 |
| Explainability | SHAP TreeExplainer |
| Train/Test Split | 80% / 20% stratified |

---

## Features

- Survive or Fail prediction with probability score
- Survival vs Failure probability bar chart
- SHAP chart explaining why the model made that prediction
- Survival Rate by Market Segment chart
- Funding vs Years Operating scatter plot
- Overall Feature Importance chart
- Confusion Matrix

---

## Input Features

| Feature | Description |
|---------|-------------|
| Funding (USD) | Total funding raised |
| Funding Rounds | Number of investment rounds |
| Team Size | Number of employees |
| Years Operating | How long the startup has been running |
| Has Revenue | Whether the startup is generating revenue |
| Pivot Count | Number of times the startup changed direction |
| Market Segment | Payments, Neobanking, Lending, Savings, Infrastructure, Agency Banking |
| CBN License | Whether the startup has CBN approval |

---

## Tech Stack

- **Language:** Python 3
- **Framework:** Streamlit
- **ML Library:** scikit-learn, XGBoost
- **Algorithm:** XGBoost Classifier
- **Explainability:** SHAP TreeExplainer
- **Data Processing:** pandas, NumPy
- **Visualisation:** Matplotlib

---

## Project Structure

```
fintech-survive-ai/
    app.py              # Main Streamlit application
    requirements.txt    # Python dependencies
    README.md           # Project documentation
```

---

## How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Santandave961/fintech-survive-ai.git
cd fintech-survive-ai
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## How It Works

1. **Data** - 60 Nigerian fintech startup data points covering survived and failed companies
2. **Encoding** - Market segment encoded using scikit-learn LabelEncoder
3. **Model** - XGBoost Classifier with 200 trees trained on a stratified 80/20 split
4. **Explainability** - SHAP TreeExplainer generates per-prediction feature contributions
5. **Output** - Survive or Fail verdict with probability scores and SHAP explanation

---

## Key Insights

- **CBN License** and **Funding** are the strongest predictors of survival
- **Payments** and **Agency Banking** segments have the highest survival rates
- Startups that pivot more than 2 times have significantly lower survival rates
- Revenue generation in the first 2 years is a strong survival signal
- Team size above 20 correlates strongly with survival

---

## Nigerian Fintechs in the Dataset

Includes data points inspired by real Nigerian fintech companies such as Kuda, Carbon, Cowrywise, PiggyVest, Flutterwave, Paystack, Moniepoint, Paga, Interswitch, Mono, Okra, and more.

---

## Author

**Okparaji Wisdom**
Data Science Student | Fintech Portfolio Builder

- GitHub: [@Santandave961](https://github.com/Santandave961)
- LinkedIn: [Connect with me](https://linkedin.com)

---

## License

MIT License - feel free to use and modify this project.
