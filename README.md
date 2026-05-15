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

## Experiments

Use `run_experiments.py` to compare models and split settings.

Examples:

```bash
python run_experiments.py --models linear random_forest gradient_boost
python run_experiments.py --models random_forest --test-size 0.25 --random-state 1 --output-dir outputs/exp_rs1_ts25
```

Results and plots are written to `outputs/` (an `experiments_results.json` is saved per run).

## Requirements

Install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Notes

- The repository previously committed `outputs/` (images); large binary outputs are usually kept out of source control. To re-commit outputs use `git add -f outputs/`.
- If you want, I can remove images from history or update `.gitignore` to keep `outputs/` untracked.
