# src/plot_residuals.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def main():
    # =========================
    # 1. データ読み込み
    # =========================
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "processed" / "sales_features.csv"

    df = pd.read_csv(data_path)

    # =========================
    # 2. 特徴量(X)と正解(y)
    # =========================
    feature_cols = [
        "is_weekend",
        "is_sale_period",
        "precip_mm",
        "on_hand_qty",
        "receipts_qty",
        "markdown_flag"
    ]

    X = df[feature_cols]
    y = df["sales_qty"]

    # =========================
    # 3. 学習 / テスト分割
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # 4. 学習
    # =========================
    model = LinearRegression()
    model.fit(X_train, y_train)

    # =========================
    # 5. 予測 & 残差
    # =========================
    y_pred = model.predict(X_test)
    residuals = y_pred - y_test

    # =========================
    # 6. 残差プロット
    # =========================
    plt.figure()
    plt.scatter(y_test, residuals)
    plt.axhline(0)  # 残差ゼロの基準線
    plt.xlabel("Actual sales_qty")
    plt.ylabel("Residual (Predicted - Actual)")
    plt.title("Residual Plot (Linear Regression)")
    plt.show()


if __name__ == "__main__":
    main()
