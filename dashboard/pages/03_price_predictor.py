import time

import streamlit as st
import pandas as pd
from data_loader import load_model, get_raw_features_df

st.set_page_config(layout="wide")
st.title("🤖 Interactive Price Predictor")
st.markdown("Select a product's features using the sidebar to get a real-time price prediction from our LightGBM model.")

# Load Model and Data
model = load_model()
df = get_raw_features_df()

st.sidebar.header("Product Features")

price_lag_1d = st.sidebar.number_input(
    "Price Yesterday (£)",
    min_value=0.0, max_value=float(df["price_lag_1d"].max()),
    value=1.50, step=0.10
)

price_rol_mean_7d = st.sidebar.slider(
    "7-Day Average Price (£)",
    min_value=0.0, max_value=float(df["price_rol_mean_7d"].max()),
    value=1.55, step=0.05
)

price_rol_max_7d = st.sidebar.slider(
    "7-Day Max Price (£)",
    min_value=0.0, max_value=float(df["price_rol_max_7d"].max()),
    value=2.00, step=0.05
)

supermarket = st.sidebar.selectbox(
    "Supermarket",
    options=['Aldi', 'ASDA', 'Morrisons', 'Sains', 'Tesco']
)

def prepare_input_data(user_input, original_columns):
    input_df = pd.DataFrame([user_input])
    # One hot encode the supermarket
    input_df["supermarket"] = supermarket
    input_df = pd.get_dummies(input_df, columns=["supermarket"])

    # We use reindex to ensure all model columns are present, filling missing ones with 0.
    final_df = input_df.reindex(columns=original_columns, fill_value=0)
    return final_df[original_columns]

# PREDICTION
if st.sidebar.button("Predict Price"): 
    user_data = {
        'price_lag_1d': price_lag_1d,
        'price_rol_mean_7d': price_rol_mean_7d,
        'price_rol_max_7d': price_rol_max_7d,
        'price_rol_min_7d': price_rol_mean_7d * 0.9,
        'price_diff_1d': price_lag_1d - (price_rol_mean_7d * 0.95),
    }

    model_features = model.feature_name_
    input_vector = prepare_input_data(user_data, model_features)

    # Timed prediction for latency display
    start = time.perf_counter()
    prediction = model.predict(input_vector)[0]
    latency_ms = (time.perf_counter() - start) * 1000

    col1, col2 = st.columns(2)
    with col1:
        st.success(f"Predicted Price: **£{prediction:.2f}**")
    with col2:
        st.info(f"⚡ Inference latency: **{latency_ms:.2f} ms**")
    st.balloons()