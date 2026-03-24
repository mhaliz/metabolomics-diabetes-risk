# Blood Screen Classifier

In a public health study, participants were enrolled from communities for physical exams, blood tests, and were followed up for years with disease outcomes.
This project aims to find relevant predictors to assess diabetes risk, and utilizes these features to build a machine learning model that will be deployed in future blood screens.

## Project Structure
    metabolomics-diabetes-risk/
    ├── data/
    │   ├── raw/                  # original data, not committed
    │   └── processed/            # train/test indices, gene column list
    ├── notebooks/
    │   ├── 01_eda.ipynb
    │   ├── 02_feature_selection.ipynb
    │   └── 03_model_selection.ipynb
    ├── src/
    │   ├── config.py             # column definitions and thresholds
    │   ├── transformers.py       # CorrelationFilter and custom transformers
    │   ├── features.py           # cv_feature_selection and helpers
    │   └── evaluate.py           # metrics and plotting helpers
    ├── models/                   # saved pipelines, not committed
    ├── tests/
    ├── requirements.txt
    └── setup.py

## Setup

### 1. Clone the repository
git clone https://github.com/mhaliz/metabolomics-diabetes-risk
cd metabolomics-diabetes-risk

### 2. Install dependencies
pip install -r requirements.txt

### 3. Install the src package in editable mode
pip install -e .

### 4. Data access
Raw data is not committed to this repository.
Place the raw file at data/raw/data.parquet before running any notebooks.

## Running the Notebooks
Run in order:
1. 01_eda.ipynb          — exploratory analysis, train/test split
2. 02_feature_selection.ipynb  — variance, correlation, univariate, lasso selection
3. 03_model_selection.ipynb    — model training and evaluation

## Requirements
- Python 3.12
- See requirements.txt for full list

## Data
This project contains data from over 8,000 participants, measuring 10,000 distinct biomarkers. It also contains demographic variables: age, sex, and BMI.

## Authors
Michali Izhaky
