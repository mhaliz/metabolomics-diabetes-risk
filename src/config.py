import pandas as pd

df = pd.read_csv("../data/raw/PUB2023_scraped.csv.gz")

GENE_COLS = [c for c in df.columns if 'mtb' in col]
SEX_COL = ['sex']
BMI_AGE_COLS = ['BMI','age']
TARGET_COLS = 'has_diabetes'

VAR_THRESHOLD = 0.1
CORR_THRESHOLD = 0.9
STABILITY_PCT = 80
