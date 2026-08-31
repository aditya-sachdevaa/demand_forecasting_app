from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

SOURCE_PATH = DATA_DIR / "processed_sales.parquet"
OUTPUT_PATH = DATA_DIR / "app_context.parquet"


def main():
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {SOURCE_PATH}. "
            "Run the data-processing notebook first."
        )

    print("Loading processed dataset...")
    df = pd.read_parquet(
        SOURCE_PATH,
        columns=[
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
            "date",
            "sales",
        ],
    )

    df["date"] = pd.to_datetime(df["date"])

    # Keep enough history to construct 28-day lag/rolling features
    # while making the Streamlit app much smaller than the full dataset.
    latest_date = df["date"].max()
    cutoff_date = latest_date - pd.Timedelta(days=90)

    context = df.loc[df["date"] >= cutoff_date].copy()

    context = context.sort_values(
        ["item_id", "store_id", "date"]
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    context.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("App context created successfully.")
    print("Rows:", len(context))
    print("Date range:", context["date"].min(), "→", context["date"].max())
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
