# src/train_linear_regression.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def main():
    # =========================
    # 1. データ読み込み
    # =========================
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "processed" / "sales_features.csv"

    df = pd.read_csv(data_path)

    # =========================
    # 2. 使う特徴量（X）と正解（y）
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
    # 3. 学習用 / テスト用に分ける
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # =========================
    # 4. モデル作成 & 学習
    # =========================
    model = LinearRegression()
    model.fit(X_train, y_train)

    # =========================
    # 5. 予測
    # =========================
    y_pred = model.predict(X_test)

    # =========================
    # 6. 評価（MAE）
    # =========================
    mae = mean_absolute_error(y_test, y_pred)

    print("✅ Linear Regression training completed")
    print(f"MAE (平均誤差): {mae:.2f}")

    # =========================
    # 7. 係数を見る（超重要）
    # =========================
    coef_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "coefficient": model.coef_
        }
    ).sort_values("coefficient", ascending=False)

    print("\n📊 Feature importance (coefficients)")
    print(coef_df)


if __name__ == "__main__":
    main()
