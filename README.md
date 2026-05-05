# Disaster Severity Prediction using Global Disaster Data

This project builds a machine learning system that predicts the expected severity of natural disaster events using historical global disaster records. The model classifies each event into one of three impact categories:

- Low Impact
- Medium Impact
- High Impact

The project demonstrates a complete tabular machine learning workflow, including data preprocessing, target variable creation, leakage prevention, feature engineering, model comparison, evaluation, explainability, prediction, deployment, and a Streamlit web interface.

---

## Live Demo

The deployed Streamlit web app is available here:

[Open Disaster Severity Prediction App](https://disaster-severity-prediction-2kpp8uxytnndavapv3pyzx.streamlit.app/)

The app allows users to enter disaster details such as country, region, disaster type, year, month, magnitude, and magnitude scale. It then predicts whether the disaster is likely to have Low, Medium, or High Impact.

---

## Project Objective

The objective of this project is to answer the question:

> Given the basic details of a disaster event, can a machine learning model classify its expected impact as low, medium, or high?

The model uses pre-impact information such as country, region, disaster type, year, month, magnitude, and magnitude scale. Final impact values such as deaths, affected population, injuries, homelessness, and economic damage are not used as input features because they directly reveal the outcome.

This makes the project practical because it moves beyond descriptive disaster analysis and focuses on prediction and decision support.

---

## Dataset

The project uses historical disaster event data from the EM-DAT International Disaster Database.

The raw dataset contains information such as:

- Country and region
- Disaster group, subgroup, type, and subtype
- Start and end dates
- Magnitude and magnitude scale
- Deaths, affected population, and economic damage

The raw dataset is not included in this repository because of dataset usage restrictions. To run the full preprocessing pipeline, place your EM-DAT Excel file here:

```text
data/raw/emdat.xlsx
```

---

## Target Variable Design

The dataset does not directly provide a clean severity label, so a custom target variable is created.

An impact score is calculated using:

```text
impact_score = log1p(total_deaths) + log1p(total_affected) + log1p(total_damage)
```

In this EM-DAT export, economic damage is taken from:

```text
total_damage_000_us
```

The impact score is then divided into three quantile-based classes:

```text
Bottom 33%  -> Low Impact
Middle 33%  -> Medium Impact
Top 33%     -> High Impact
```

This creates a balanced multi-class classification problem and avoids manually choosing arbitrary thresholds.

---

## Data Leakage Prevention

A major part of this project is preventing data leakage.

The following columns are used only for target creation and are removed before training:

- Total deaths
- Injured
- Affected population
- Homeless population
- Total affected
- Reconstruction costs
- Insured damage
- Total damage
- Impact score

This ensures that the model predicts severity from information that could be known before the final disaster impact is fully reported.

---

## Features Used for Training

The final model uses the following input features:

- Country
- Region
- Disaster group
- Disaster subgroup
- Disaster type
- Disaster subtype
- Start year
- Start month
- End year
- Magnitude
- Magnitude scale
- Decade
- Disaster duration in years
- Country disaster frequency
- Disaster type frequency
- Magnitude missing flag

---

## Feature Engineering

The following engineered features were added:

| Feature | Description |
|---|---|
| Decade | Converts year into decade groups such as 1990s, 2000s, and 2010s |
| Disaster duration | Difference between end year and start year |
| Country disaster frequency | Number of historical records for each country |
| Disaster type frequency | Number of historical records for each disaster type |
| Magnitude missing flag | Indicates whether magnitude was missing |

These features help the model capture historical, geographical, and disaster-type patterns.

---

## Models Trained

The project compares multiple models instead of directly choosing a final model.

| Model | Purpose |
|---|---|
| Logistic Regression | Simple baseline model |
| Decision Tree | Interpretable baseline |
| Random Forest | Strong non-linear tabular model |
| XGBoost | Gradient boosting model |
| LightGBM | Best-performing final model |

---

## Model Performance

The best-performing model was **LightGBM**.

| Model | Accuracy | Macro F1-score | Weighted F1-score |
|---|---:|---:|---:|
| LightGBM | 0.5662 | 0.5657 | 0.5660 |
| XGBoost | 0.5625 | 0.5613 | 0.5618 |
| Random Forest | 0.5597 | 0.5583 | 0.5587 |
| Logistic Regression | 0.5305 | 0.5293 | 0.5297 |
| Decision Tree | 0.4754 | 0.4752 | 0.4755 |

Macro F1-score is used as the main comparison metric because this is a multi-class classification problem and each severity class should be treated fairly.

---

## Final Model

The final selected model is:

```text
LightGBM
```

The trained model is saved as:

```text
models/best_model.joblib
```

The label encoder is saved as:

```text
models/label_encoder.joblib
```

---

## Feature Importance

The final LightGBM model found the following features to be highly important:

- Start year
- Country disaster frequency
- Start month
- Magnitude
- End year
- Disaster type frequency
- Region
- Magnitude scale
- Disaster subgroup
- Disaster subtype
- Decade

Feature importance outputs are saved in:

```text
reports/feature_importance.csv
reports/feature_importance.png
```

---

## Streamlit Web App

A Streamlit web app is included to make the model interactive and easy to test.

The app file is:

```text
app/streamlit_app.py
```

To run the app locally:

```bash
streamlit run app/streamlit_app.py
```

The app takes user inputs such as:

- Country
- Region
- Disaster group
- Disaster subgroup
- Disaster type
- Disaster subtype
- Start year
- Start month
- End year
- Magnitude
- Magnitude scale

It returns:

- Predicted severity class
- Class probabilities
- Probability bar chart

The deployed version of the app can be accessed here:

[Open Disaster Severity Prediction App](https://disaster-severity-prediction-2kpp8uxytnndavapv3pyzx.streamlit.app/)

> Replace the placeholder link with your actual deployed Streamlit URL.

---

## Sample Prediction Output

```text
Sample Disaster Event Prediction
============================================================
Predicted severity: High Impact

Class probabilities:
  severity_class  probability
0    High Impact     0.673125
2  Medium Impact     0.238141
1     Low Impact     0.088734
```

---

## Repository Structure

```text
disaster-severity-prediction/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
│       └── cleaned_disasters.csv
├── models/
│   ├── best_model.joblib
│   └── label_encoder.joblib
├── reports/
│   ├── model_comparison.csv
│   ├── final_model_metrics.txt
│   ├── confusion_matrix.png
│   ├── feature_importance.csv
│   └── feature_importance.png
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate.py
│   ├── feature_importance.py
│   └── predict.py
├── .gitignore
├── README.md
├── requirements.txt
└── runtime.txt
```

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Dexter087/disaster-severity-prediction.git
cd disaster-severity-prediction
```

### 2. Create a Conda environment

```bash
conda create -n disasterml python=3.10 -y
conda activate disasterml
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Full Pipeline

### 1. Add the raw dataset

Place the EM-DAT Excel file here:

```text
data/raw/emdat.xlsx
```

### 2. Preprocess the dataset

```bash
python src/preprocess.py
```

This creates:

```text
data/processed/cleaned_disasters.csv
```

### 3. Train the models

```bash
python src/train_model.py
```

This trains Logistic Regression, Decision Tree, Random Forest, XGBoost, and LightGBM.

It creates:

```text
models/best_model.joblib
models/label_encoder.joblib
reports/model_comparison.csv
```

### 4. Evaluate the final model

```bash
python src/evaluate.py
```

This creates:

```text
reports/final_model_metrics.txt
reports/confusion_matrix.png
```

### 5. Generate feature importance

```bash
python src/feature_importance.py
```

This creates:

```text
reports/feature_importance.csv
reports/feature_importance.png
```

### 6. Run a sample prediction

```bash
python src/predict.py
```

### 7. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

---

## Important Deployment Fix

The saved model depends on the same package versions used during training. If the deployed app uses a different version of Python or scikit-learn, it may fail with an error such as:

```text
AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'
```

To avoid this, the project uses pinned dependencies.

The `runtime.txt` file should contain:

```text
python-3.10
```

## Key Skills Demonstrated

- Real-world disaster data preprocessing
- Missing value handling
- Feature engineering
- Custom target variable design
- Data leakage prevention
- Multi-class classification
- Model comparison
- Gradient boosting models
- LightGBM training
- Feature importance analysis
- Saved ML pipeline
- Streamlit app development
- Cloud deployment preparation

---

## Future Improvements

Possible future improvements include:

- Add World Bank population and GDP data
- Add country income group as an external feature
- Use climate-related indicators
- Tune LightGBM hyperparameters
- Add SHAP explainability
- Add more interactive visualizations to the Streamlit app
- Use a time-based train-test split
- Add batch CSV prediction support
- Improve handling of unknown countries and disaster types

---
