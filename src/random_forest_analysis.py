# src/random_forest_analysis.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


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
    # 4. RandomForest 学習
    # =========================
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # =========================
    # 5. 予測
    # =========================
    y_pred = model.predict(X_test)

    # =========================
    # 6. MAE
    # =========================
    mae = mean_absolute_error(y_test, y_pred)
    print(f"🌲 RandomForest MAE: {mae:.2f}")

    # =========================
    # 7. 散布図（Actual vs Predicted）
    # =========================
    plt.figure()
    plt.scatter(y_test, y_pred)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val])
    plt.xlabel("Actual sales_qty")
    plt.ylabel("Predicted sales_qty")
    plt.title("Actual vs Predicted (RandomForest)")
    plt.show()

    # =========================
    # 8. 残差プロット
    # =========================
    residuals = y_pred - y_test
    plt.figure()
    plt.scatter(y_test, residuals)
    plt.axhline(0)
    plt.xlabel("Actual sales_qty")
    plt.ylabel("Residual (Predicted - Actual)")
    plt.title("Residual Plot (RandomForest)")
    plt.show()


if __name__ == "__main__":
    main()
