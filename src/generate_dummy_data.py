# src/generate_dummy_data.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def main() -> None:
    # ==========
    # 設定（ここだけ変えればOK）
    # ==========
    STORE_ID = "store_001"
    START_DATE = "2026-01-01"     # 90日をここから作る
    N_DAYS = 90
    N_SKUS = 10
    SEED = 42

    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    # ==========
    # 1) calendar.csv（90日）
    # ==========
    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")
    cal = pd.DataFrame({"date": dates.date})

    # dow: 0=Mon ... 6=Sun（pandasの曜日）
    cal["dow"] = dates.weekday
    cal["is_weekend"] = (cal["dow"] >= 5).astype(int)

    # 祝日は今回はダミー（面接で「将来差し替え」と言える）
    cal["is_holiday"] = 0
    cal["holiday_name"] = ""

    # 給料日も簡易（25日を給料日にしてみる）
    cal["is_payday"] = (dates.day == 25).astype(int)

    # セール期間：毎月最初の7日をセール扱い（簡易でOK）
    cal["is_sale_period"] = (dates.day <= 7).astype(int)

    # season: ざっくり月で分類（日本向け）
    month = dates.month
    cal["season"] = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8]), month.isin([9, 10, 11])],
        ["winter", "spring", "summer", "autumn"],
        default="unknown",
    )

    cal["month"] = dates.month
    cal["week_of_year"] = dates.isocalendar().week.astype(int)

    # ==========
    # 2) weather_daily.csv（90日×1店舗）
    # ==========
    # 気温は季節っぽく「ゆるい波＋ノイズ」
    t = np.arange(N_DAYS)
    temp_avg = 10 + 4 * np.sin(2 * np.pi * t / 30) + rng.normal(0, 1.2, N_DAYS)

    # 降水：30%くらいの確率で雨（雨量はガンマ分布でそれっぽく）
    is_rain = rng.random(N_DAYS) < 0.30
    precip = np.where(is_rain, rng.gamma(shape=2.0, scale=3.0, size=N_DAYS), 0.0)

    weather = pd.DataFrame(
        {
            "date": dates.date,
            "store_id": STORE_ID,
            "temp_avg_c": np.round(temp_avg, 1),
            "temp_max_c": np.round(temp_avg + rng.uniform(1.0, 5.0, N_DAYS), 1),
            "temp_min_c": np.round(temp_avg - rng.uniform(1.0, 5.0, N_DAYS), 1),
            "precip_mm": np.round(precip, 1),
            "weather": np.where(precip > 0, "rain", "clear"),
            "humidity_avg": np.round(rng.uniform(40, 75, N_DAYS), 0),
            "wind_avg_mps": np.round(rng.uniform(0.5, 5.5, N_DAYS), 1),
        }
    )

    # ==========
    # 3) sales_daily.csv（90日×10品番）
    # ==========
    # 10品番（Levi'sっぽい例）
    product_codes = [501, 502, 505, 511, 512, 514, 517, 550, 565, 578][:N_SKUS]
    fits = ["Straight", "Taper", "Regular", "Slim", "Slim", "Straight", "Bootcut", "Relaxed", "Loose", "Loose"][:N_SKUS]
    categories = ["Denim"] * N_SKUS
    msrps = [15400, 15400, 14300, 15400, 15400, 14300, 14300, 13200, 14300, 15400][:N_SKUS]

    sku_master = pd.DataFrame(
        {
            "sku_id": [f"sku_{i+1:03d}" for i in range(N_SKUS)],
            "product_code": product_codes,
            "fit": fits,
            "category": categories,
            "msrp_yen": msrps,
        }
    )

    # date × sku の全組み合わせ（900行）
    base = pd.MultiIndex.from_product([dates.date, sku_master["sku_id"]], names=["date", "sku_id"]).to_frame(index=False)
    sales = base.merge(sku_master, on="sku_id", how="left")
    sales.insert(1, "store_id", STORE_ID)

    # --- 需要（売れやすさ）を作る（ルール付きダミー） ---
    # ベース需要：品番ごとの差（少しだけ）
    sku_effect = rng.normal(loc=0.0, scale=0.25, size=N_SKUS)
    sku_effect_map = dict(zip(sku_master["sku_id"], sku_effect))

    # calendar要因を付与
    cal_map = cal.set_index("date")
    sales["dow"] = sales["date"].map(cal_map["dow"])
    sales["is_weekend"] = sales["date"].map(cal_map["is_weekend"])
    sales["is_sale_period"] = sales["date"].map(cal_map["is_sale_period"])

    # weather要因を付与（雨の日は少しマイナス）
    w_map = weather.set_index("date")
    sales["precip_mm"] = sales["date"].map(w_map["precip_mm"])

    # 期待販売数（lambda）を作る：Poissonで自然な整数に
    # 週末 +25%、セール +35%、雨 -15%（ゆるい仮説）
    base_lambda = 8.0
    weekend_mult = np.where(sales["is_weekend"] == 1, 1.25, 1.00)
    sale_mult = np.where(sales["is_sale_period"] == 1, 1.35, 1.00)
    rain_mult = np.where(sales["precip_mm"] > 0, 0.85, 1.00)
    sku_mult = sales["sku_id"].map(lambda x: np.exp(sku_effect_map[x]))  # 0.7〜1.4くらいに散らす

    lam = base_lambda * weekend_mult * sale_mult * rain_mult * sku_mult
    lam = np.clip(lam, 0.1, None)

    # 販売数（まずは「欠品考慮前」）
    sales_qty_raw = rng.poisson(lam=lam)

    # 在庫（on_hand）と入荷（receipts）をそれっぽく
    # 在庫が少ない日を混ぜる（欠品発生のため）
    on_hand = rng.integers(low=0, high=60, size=len(sales))
    receipts = rng.integers(low=0, high=20, size=len(sales))

    # 欠品フラグ：在庫0なら欠品
    stockout_flag = (on_hand == 0).astype(int)

    # 欠品なら売上は0、欠品じゃない場合も在庫以上は売れない
    sales_qty = np.where(stockout_flag == 1, 0, np.minimum(sales_qty_raw, on_hand))

    # 値引き（markdown）：セール期間の一部を1に（簡易）
    markdown_flag = sales["is_sale_period"].astype(int)

    # 売上金額：定価×数量×(値引き)
    # 値引き時は 20%OFF くらいに
    discount = np.where(markdown_flag == 1, 0.80, 1.00)
    net_sales_yen = np.round(sales["msrp_yen"] * sales_qty * discount).astype(int)

    # 仕上げ：不要な中間列は消す（rawのスキーマに合わせる）
    sales_out = sales.copy()
    sales_out["sales_qty"] = sales_qty.astype(int)
    sales_out["net_sales_yen"] = net_sales_yen
    sales_out["on_hand_qty"] = on_hand.astype(int)
    sales_out["receipts_qty"] = receipts.astype(int)
    sales_out["stockout_flag"] = stockout_flag.astype(int)
    sales_out["markdown_flag"] = markdown_flag.astype(int)

    sales_out = sales_out[
        [
            "date",
            "store_id",
            "sku_id",
            "product_code",
            "fit",
            "category",
            "msrp_yen",
            "sales_qty",
            "net_sales_yen",
            "on_hand_qty",
            "receipts_qty",
            "stockout_flag",
            "markdown_flag",
        ]
    ]

    # ==========
    # CSV書き出し
    # ==========
    sales_out.to_csv(raw_dir / "sales_daily.csv", index=False, encoding="utf-8")
    cal.to_csv(raw_dir / "calendar.csv", index=False, encoding="utf-8")
    weather.to_csv(raw_dir / "weather_daily.csv", index=False, encoding="utf-8")

    print("✅ Dummy data generated:")
    print(f" - {raw_dir / 'sales_daily.csv'}  ({len(sales_out)} rows)")
    print(f" - {raw_dir / 'calendar.csv'}     ({len(cal)} rows)")
    print(f" - {raw_dir / 'weather_daily.csv'} ({len(weather)} rows)")


if __name__ == "__main__":
    main()
