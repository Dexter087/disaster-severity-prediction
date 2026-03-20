# Disaster Severity Prediction using Global Disaster Data

This project builds a machine learning system that predicts the severity level of natural disaster events using historical disaster data. The model classifies each disaster into one of three impact categories:

- Low Impact
- Medium Impact
- High Impact

The project uses real-world disaster records and demonstrates data cleaning, feature engineering, leakage prevention, model comparison, evaluation, and explainable machine learning.

## Project Objective

The goal of this project is to answer the question:

> Given the basic details of a disaster event, can a machine learning model classify its expected impact as low, medium, or high?

The model uses information such as country, region, disaster type, start year, start month, magnitude, and magnitude scale. Final impact-related values such as total deaths, affected population, and economic damage are used only to create the target variable and are not used as model input features.

## Dataset

The project uses disaster event records from the EM-DAT International Disaster Database.

The dataset contains information such as:

- Country and region
- Disaster group, type, and subtype
- Start and end dates
- Magnitude and magnitude scale
- Deaths, affected population, and damage values

The raw dataset is not included in this repository because of data usage restrictions. Users should place their EM-DAT export file inside:

```text
data/raw/emdat.xlsx