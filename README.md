# Predicting-Insurance-Claim-Amounts

Estimate the medical insurance claim amount based on personal data.

## Objective
Train a **Linear Regression** model to predict medical insurance `charges`, visualize the impact of key features, and evaluate model performance with **MAE** and **RMSE**.

## Dataset
The script uses the Medical Cost Personal Dataset:
- https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv

## Requirements
Install dependencies:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

## Run

```bash
python insurance_claim_prediction.py
```

## Output
The script prints:
- MAE
- RMSE

The script also saves visualizations in `outputs/`:
- `bmi_vs_charges.png`
- `age_vs_charges.png`
- `smoker_vs_charges.png`
