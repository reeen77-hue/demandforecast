# src/preprocess.py
from pathlib import Path
import pandas as pd


def main():
    # =========================
    # パス設定
    # =========================
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # 1. CSV読み込み
    # =========================
    sales = pd.read_csv(raw_dir / "sales_daily.csv")
    calendar = pd.read_csv(raw_dir / "calendar.csv")
    weather = pd.read_csv(raw_dir / "weather_daily.csv")

    # =========================
    # 2. 型の整理（日付）
    # =========================
    sales["date"] = pd.to_datetime(sales["date"])
    calendar["date"] = pd.to_datetime(calendar["date"])
    weather["date"] = pd.to_datetime(weather["date"])

    # =========================
    # 3. calendar を結合（date）
    # =========================
    df = sales.merge(
        calendar,
        on="date",
        how="left"
    )

    # =========================
    # 4. weather を結合（date + store_id）
    # =========================
    df = df.merge(
        weather,
        on=["date", "store_id"],
        how="left"
    )

    # =========================
    # 5. 学習用に列を整理
    # =========================
    # 今回は説明用の列は落とす
    drop_cols = [
        "holiday_name",   # 説明用
        "weather"         # 文字列なので今回は使わない
    ]
    df = df.drop(columns=drop_cols)

    # =========================
    # 6. 欠損値チェック（最低限）
    # =========================
    # フラグ系は欠損があれば 0 に
    flag_cols = [
        "is_weekend",
        "is_holiday",
        "is_payday",
        "is_sale_period",
        "stockout_flag",
        "markdown_flag"
    ]
    for col in flag_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # =========================
    # 7. 保存
    # =========================
    output_path = processed_dir / "sales_features.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")

    print("✅ Preprocess completed")
    print(f"Saved to: {output_path}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
