from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

DATASET_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"


def load_dataset() -> pd.DataFrame:
    """Load the medical cost personal dataset."""
    return pd.read_csv(DATASET_URL)


def train_and_evaluate(data: pd.DataFrame) -> tuple[LinearRegression, pd.DataFrame, dict[str, float]]:
    """Train a linear regression model and return evaluation metrics."""
    encoded = pd.get_dummies(data, columns=["sex", "smoker", "region"], drop_first=True)
    X = encoded.drop(columns=["charges"])
    y = encoded["charges"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions) ** 0.5,
    }
    return model, data, metrics


def create_visualizations(data: pd.DataFrame, output_dir: Path) -> None:
    """Create plots showing feature impact on insurance charges."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=data, x="bmi", y="charges", hue="smoker", alpha=0.7)
    plt.title("BMI vs Insurance Charges")
    plt.tight_layout()
    plt.savefig(output_dir / "bmi_vs_charges.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=data, x="age", y="charges", hue="smoker", alpha=0.7)
    plt.title("Age vs Insurance Charges")
    plt.tight_layout()
    plt.savefig(output_dir / "age_vs_charges.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.boxplot(data=data, x="smoker", y="charges")
    plt.title("Smoking Status vs Insurance Charges")
    plt.tight_layout()
    plt.savefig(output_dir / "smoker_vs_charges.png", dpi=150)
    plt.close()


def main() -> None:
    data = load_dataset()
    _, original_data, metrics = train_and_evaluate(data)
    create_visualizations(original_data, Path("outputs"))

    print("Linear Regression performance on insurance charges")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print("Saved plots to outputs/: bmi_vs_charges.png, age_vs_charges.png, smoker_vs_charges.png")


if __name__ == "__main__":
    main()
