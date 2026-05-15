from pathlib import Path
import argparse
import json

import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

DATASET_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_URL)


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    encoded = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)
    X = encoded.drop(columns=["charges"])
    y = encoded["charges"]
    return X, y


def make_model(name: str):
    if name == "linear":
        return LinearRegression()
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=100, random_state=0)
    if name == "gradient_boost":
        return GradientBoostingRegressor(n_estimators=100, random_state=0)
    raise ValueError(f"Unknown model: {name}")


def evaluate_model(model, X_train, X_test, y_train, y_test) -> dict:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(mean_squared_error(y_test, preds) ** 0.5),
    }


def run(models, test_size, random_state, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    results = {}
    for name in models:
        model = make_model(name)
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        results[name] = metrics
        print(f"{name}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")

    (output_dir / "experiments_results.json").write_text(json.dumps(results, indent=2))
    # create a simple plot for BMI vs charges
    sns.set_theme(style="whitegrid")
    plt = __import__("matplotlib.pyplot").pyplot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="bmi", y="charges", hue="smoker", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "bmi_vs_charges_experiment.png", dpi=150)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["linear", "random_forest", "gradient_boost"]) 
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


def main():
    args = parse_args()
    run(args.models, args.test_size, args.random_state, args.output_dir)


if __name__ == "__main__":
    main()
