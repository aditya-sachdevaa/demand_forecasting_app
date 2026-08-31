from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

APP_CONTEXT_PATH = DATA_DIR / "app_context.parquet"
CALENDAR_PATH = DATA_DIR / "raw" / "m5-forecasting-accuracy" / "calendar.csv"
SELL_PRICES_PATH = DATA_DIR / "raw" / "m5-forecasting-accuracy" / "sell_prices.csv"

MODEL_PATH = MODELS_DIR / "final_lightgbm_model.pkl"
FEATURES_PATH = MODELS_DIR / "final_feature_cols.pkl"


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Demand Forecasting",
    page_icon="📦",
    layout="wide",
)


# ------------------------------------------------------------
# Loading helpers
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_list():
    return joblib.load(FEATURES_PATH)


@st.cache_data
def load_context():
    if not APP_CONTEXT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {APP_CONTEXT_PATH.name}. "
            "Run prepare_app_data.py once before starting the app."
        )

    df = pd.read_parquet(APP_CONTEXT_PATH)
    df["date"] = pd.to_datetime(df["date"])

    categorical_cols = [
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "event_type",
        "event_name",
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


@st.cache_data
def load_calendar():
    calendar = pd.read_csv(CALENDAR_PATH)
    calendar["date"] = pd.to_datetime(calendar["date"])

    for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        calendar[col] = calendar[col].fillna("None")

    return calendar


@st.cache_data
def load_sell_prices():
    prices = pd.read_csv(SELL_PRICES_PATH)

    prices["sell_price"] = prices["sell_price"].astype("float32")

    return prices


# ------------------------------------------------------------
# Feature construction
# ------------------------------------------------------------
def get_target_calendar_row(calendar: pd.DataFrame, target_date: pd.Timestamp):
    row = calendar.loc[calendar["date"] == target_date]

    if row.empty:
        raise ValueError(
            f"No calendar information is available for {target_date.date()}."
        )

    return row.iloc[0]


def get_target_price(
    prices: pd.DataFrame,
    item_id: str,
    store_id: str,
    wm_yr_wk: int,
):
    row = prices.loc[
        (prices["item_id"] == item_id)
        & (prices["store_id"] == store_id)
        & (prices["wm_yr_wk"] == wm_yr_wk)
    ]

    if row.empty:
        return 0.0, 1

    return float(row["sell_price"].iloc[0]), 0


def build_features(
    context: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    item_id: str,
    store_id: str,
    target_date: pd.Timestamp,
):
    series = context.loc[
        (context["item_id"] == item_id)
        & (context["store_id"] == store_id)
        & (context["date"] < target_date)
    ].sort_values("date")

    if series.empty:
        raise ValueError("No historical sales found for this item/store pair.")

    latest = series.iloc[-1]

    required_history = 28
    if len(series) < required_history:
        raise ValueError(
            f"Not enough history to construct lag/rolling features. "
            f"Need at least {required_history} previous observations."
        )

    calendar_row = get_target_calendar_row(calendar, target_date)

    sell_price, price_missing = get_target_price(
        prices=prices,
        item_id=item_id,
        store_id=store_id,
        wm_yr_wk=int(calendar_row["wm_yr_wk"]),
    )

    sales = series["sales"].to_numpy(dtype=float)

    features = {
        "item_id": item_id,
        "dept_id": latest["dept_id"],
        "cat_id": latest["cat_id"],
        "store_id": store_id,
        "state_id": latest["state_id"],
        "wday": float(calendar_row["wday"]),
        "month": float(calendar_row["month"]),
        "year": float(calendar_row["year"]),
        "has_event": int(
            (
                calendar_row["event_name_1"] != "None"
                or calendar_row["event_name_2"] != "None"
            )
        ),
        "event_type": (
            calendar_row["event_type_1"]
            if calendar_row["event_type_1"] != "None"
            else calendar_row["event_type_2"]
        ),
        "event_name": (
            calendar_row["event_name_1"]
            if calendar_row["event_name_1"] != "None"
            else calendar_row["event_name_2"]
        ),
        "snap": float(
            calendar_row[f"snap_{latest['state_id']}"]
        ),
        "sell_price": sell_price,
        "price_missing": price_missing,
        "lag_1": sales[-1],
        "lag_2": sales[-2],
        "lag_3": sales[-3],
        "lag_7": sales[-7],
        "lag_14": sales[-14],
        "lag_28": sales[-28],
        "rolling_mean_7": sales[-7:].mean(),
        "rolling_mean_14": sales[-14:].mean(),
        "rolling_mean_28": sales[-28:].mean(),
    }

    return pd.DataFrame([features])


def make_prediction(
    model,
    feature_cols,
    context,
    calendar,
    prices,
    item_id,
    store_id,
    target_date,
):
    X = build_features(
        context=context,
        calendar=calendar,
        prices=prices,
        item_id=item_id,
        store_id=store_id,
        target_date=target_date,
    )

    categorical_cols = [
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "event_type",
        "event_name",
    ]

    # Match the categorical levels used when the LightGBM model was trained.
    # This keeps category codes consistent between training and inference.
    pandas_categorical = getattr(model.booster_, "pandas_categorical", None)

    if pandas_categorical is not None:
        for col, categories in zip(categorical_cols, pandas_categorical):
            X[col] = pd.Categorical(
                X[col],
                categories=categories
            )
    else:
        for col in categorical_cols:
            X[col] = X[col].astype("category")

    X = X[feature_cols]

    raw_prediction = float(model.predict(X)[0])

    # Sales cannot be negative. Keep the raw model output internally,
    # but never display a negative demand forecast to the user.
    prediction = max(0.0, raw_prediction)

    return prediction, raw_prediction, X


# ------------------------------------------------------------
# App
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(148, 163, 184, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 10px;
        padding: 0.9rem 1rem;
    }
    div[data-testid="stExpander"] {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📦 Demand Forecasting")
st.write(
    "A simple item-store demand forecasting tool powered by the final "
    "LightGBM model from the project."
)
st.caption(
    "The app uses historical demand, recent sales patterns, calendar information, "
    "events, SNAP indicators, pricing, and item/store context."
)

if not APP_CONTEXT_PATH.exists():
    st.error(
        "The app context file is missing. Run `prepare_app_data.py` once, "
        "then restart Streamlit."
    )
    st.stop()

try:
    model = load_model()
    feature_cols = load_feature_list()
    context = load_context()
    calendar = load_calendar()
    prices = load_sell_prices()
except Exception as exc:
    st.error(f"Could not load the forecasting assets: {exc}")
    st.stop()


# Available series
series_index = (
    context[["item_id", "store_id"]]
    .drop_duplicates()
    .sort_values(["item_id", "store_id"])
    .reset_index(drop=True)
)

items = sorted(series_index["item_id"].unique().tolist())

if "selected_item" not in st.session_state:
    st.session_state.selected_item = None


# ------------------------------------------------------------
# Sidebar: item search + series selection
# ------------------------------------------------------------
with st.sidebar:
    st.header("🔍 Choose a series")

    # A single searchable dropdown: click it (or start typing) and a
    # filtered list of matching items appears underneath, just like a
    # browser address bar or a modern app's search box.
    selected_item = st.selectbox(
        "Search for an item",
        options=items,
        index=items.index(st.session_state.selected_item)
        if st.session_state.selected_item in items
        else None,
        placeholder="Type any part of the item ID, e.g. 090 or FOODS_3_090",
        help="Start typing to filter the list, then pick a result from the dropdown.",
    )

    st.session_state.selected_item = selected_item

if st.session_state.selected_item is None:
    st.info("👈 Search for an item in the sidebar to get started.")
    st.stop()

item_id = st.session_state.selected_item

stores_for_item = (
    series_index.loc[
        series_index["item_id"] == item_id,
        "store_id"
    ]
    .unique()
    .tolist()
)

with st.sidebar:
    st.divider()
    st.subheader("Store & date")

    store_id = st.selectbox(
        "Select store",
        stores_for_item,
    )

    min_date = context["date"].min().date()
    max_date = context["date"].max().date()

    default_date = max_date + pd.Timedelta(days=1)

    forecast_date = st.date_input(
        "Forecast date",
        value=default_date,
        min_value=min_date + pd.Timedelta(days=28),
        max_value=max_date + pd.Timedelta(days=28),
    )

    forecast_date = pd.Timestamp(forecast_date)

    st.divider()
    generate_clicked = st.button(
        "Generate Forecast", type="primary", use_container_width=True
    )


# ------------------------------------------------------------
# Main area: summary + results
# ------------------------------------------------------------
summary_cols = st.columns(3)
summary_cols[0].markdown(f"**Item**  \n`{item_id}`")
summary_cols[1].markdown(f"**Store**  \n`{store_id}`")
summary_cols[2].markdown(f"**Forecast date**  \n`{forecast_date.date()}`")

st.divider()

if generate_clicked:
    try:
        with st.spinner("Generating forecast..."):
            prediction, raw_prediction, feature_row = make_prediction(
                model=model,
                feature_cols=feature_cols,
                context=context,
                calendar=calendar,
                prices=prices,
                item_id=item_id,
                store_id=store_id,
                target_date=forecast_date,
            )

            latest_history = (
                context.loc[
                    (context["item_id"] == item_id)
                    & (context["store_id"] == store_id)
                    & (context["date"] < forecast_date)
                ]
                .sort_values("date")
                .tail(28)
            )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Predicted demand",
            f"{prediction:.1f} units",
        )

        col2.metric(
            "Recent 7-day average",
            f"{feature_row['rolling_mean_7'].iloc[0]:.1f} units",
        )

        col3.metric(
            "Previous day sales",
            f"{feature_row['lag_1'].iloc[0]:.0f} units",
        )

        st.subheader("Forecast")

        forecast_event = feature_row["event_name"].iloc[0]
        forecast_price = feature_row["sell_price"].iloc[0]

        info_cols = st.columns(4)
        info_cols[0].write(f"**Item:** {item_id}")
        info_cols[1].write(f"**Store:** {store_id}")
        info_cols[2].write(f"**Date:** {forecast_date.date()}")
        info_cols[3].write(
            f"**Event:** {'Yes' if forecast_event != 'None' else 'No'}"
        )

        st.write(
            f"The model forecasts **{prediction:.1f} units** of demand for "
            f"**{item_id}** at **{store_id}** on **{forecast_date.date()}**."
        )

        st.caption(
            f"Expected selling price used by the model: {forecast_price:.2f}"
        )

        st.subheader("Recent demand history")

        chart_data = latest_history[["date", "sales"]].copy()
        chart_data = chart_data.set_index("date")
        chart_data = chart_data.rename(columns={"sales": "Units sold"})

        st.line_chart(chart_data)

        # Historical comparison is available when the selected forecast date
        # falls inside the compact context dataset.
        actual_match = context.loc[
            (context["item_id"] == item_id)
            & (context["store_id"] == store_id)
            & (context["date"] == forecast_date)
        ]

        if not actual_match.empty:
            actual = float(actual_match["sales"].iloc[0])
            error = abs(actual - prediction)

            st.subheader("Historical check")

            check_cols = st.columns(3)
            check_cols[0].metric("Predicted", f"{prediction:.1f} units")
            check_cols[1].metric("Actual", f"{actual:.0f} units")
            check_cols[2].metric("Absolute error", f"{error:.1f} units")

            st.success(
                "Because this date is historical, the forecast can be compared "
                "directly with the observed demand."
            )

        elif raw_prediction < 0:
            st.info(
                "The underlying regression output was slightly negative, so "
                "the displayed forecast was constrained to zero."
            )

    except Exception as exc:
        st.error(f"Could not generate the forecast: {exc}")
else:
    st.info("Set your store and forecast date in the sidebar, then click **Generate Forecast**.")


st.divider()

with st.expander("Model performance"):
    metric_cols = st.columns(2)

    metric_cols[0].metric("Test MAE", "0.91")
    metric_cols[1].metric("Test RMSE", "2.02")

    st.write(
        "On the unseen 28-day test period, the final model reduced MAE by "
        "approximately 24% compared with the lag-1 baseline."
    )

with st.expander("About the model"):
    st.write(
        "The final model is a LightGBM regression model using item/store "
        "identity, calendar variables, price information, event/SNAP indicators, "
        "recent sales lags, and 7/14/28-day rolling demand averages."
    )

with st.expander("Known limitation"):
    st.write(
        "The model is strongest on typical low-to-moderate demand and tends "
        "to underestimate unusually high-demand observations. This is the "
        "main area identified for future improvement."
    )