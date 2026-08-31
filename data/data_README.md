# Data

This project uses the **M5 Forecasting - Accuracy** dataset.

The original M5 data is not included in this GitHub repository because the raw and processed files are large. Download the dataset separately and place the required files in the locations below.

## Required raw files

Place these files inside:

```text
data/raw/
```

### 1. `sales_train_evaluation.csv`

Contains the historical daily unit sales for each item-store series.

### 2. `calendar.csv`

Contains the date-level calendar information, including weekday information, events, and SNAP indicators.

### 3. `sell_prices.csv`

Contains weekly selling prices for item-store combinations.

## Expected local structure

```text
demand-forecasting/
└── data/
    ├── README.md
    └── raw/
        ├── sales_train_evaluation.csv
        ├── calendar.csv
        └── sell_prices.csv
```

## Processed data

The data-processing notebook creates the feature-engineered dataset used for model training.

Because the processed file is also large, it is intentionally excluded from GitHub.

The local project will eventually contain:

```text
data/
├── raw/
│   ├── sales_train_evaluation.csv
│   ├── calendar.csv
│   └── sell_prices.csv
│
├── processed_sales.parquet
└── app_context.parquet
```

`processed_sales.parquet` is the main feature-engineered dataset.

`app_context.parquet` is a smaller context file created by `prepare_app_data.py` so that the Streamlit application does not need to load the complete processed dataset every time.

## Source

M5 Forecasting - Accuracy:

https://www.kaggle.com/competitions/m5-forecasting-accuracy
