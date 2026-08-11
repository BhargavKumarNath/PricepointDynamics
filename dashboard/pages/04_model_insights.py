import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from data_loader import load_model, load_shap_sample_data, load_shap_values
from utils import set_plot_style

# Page Configuration
st.set_page_config(page_title="Model Insights", layout="wide")

st.markdown("<h1 style='text-align: center; color: white;'>🧠 Model Insights & Explainable AI (XAI)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>This page delves into the 'brain' of our price prediction model. We use SHAP to understand not just *what* the model predicts, but *why*.</p>", unsafe_allow_html=True)
st.divider()

# Load Model
model = load_model()

# Load PRE-COMPUTED SHAP data
with st.spinner("Loading pre-computed SHAP analysis..."):
    X_sample = load_shap_sample_data()
    shap_values_sample, base_value = load_shap_values()

# Check if SHAP data loaded successfully
if X_sample is None or shap_values_sample is None:
    st.error("""
    ### Pre-computed SHAP data not available
    
    To enable SHAP analysis, run:
    ```bash
    python run.py precompute
    ```
    """)
    st.stop()

st.success(f"✓ SHAP analysis loaded for {len(X_sample):,} samples!")

# Model Performance
with st.container(border=True):
    st.subheader("Model Performance")
    st.markdown("The LightGBM model was trained on over 4.9 million data points and tested on a hold-out set of the final week's data (~460k records).")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Absolute Error (MAE)", "£0.14", help="On average, the model's prediction is off by 14 pence.")
    col2.metric("Root Mean Squared Error (RMSE)", "£0.37", help="Penalizes larger errors more heavily.")
    col3.metric("Dataset Size", "9.5M+", help="Total records analyzed in the project.")

st.divider()

# Global Feature Importance
with st.container(border=True):
    st.subheader("Global Feature Importance: What Drives Prices Overall?")
    st.markdown("The plots below show which features have the most predictive power across the entire dataset.")
    
    tab1, tab2 = st.tabs(["Bar Chart (Average Impact)", "Beeswarm Plot (Impact Distribution)"])

    with tab1:
        set_plot_style()
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values_sample, X_sample, plot_type="bar", show=False, max_display=15)
        
        ax.set_xlabel("mean(|SHAP value|) (average impact on model output)", color="white", fontsize=11)
        ax.tick_params(axis='x', colors='white', labelsize=10)
        ax.tick_params(axis='y', colors='white', labelsize=10)
        
        plt.tight_layout()
        st.pyplot(fig, width='stretch')

    with tab2:
        set_plot_style()
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values_sample, X_sample, show=False, max_display=15)
        
        ax2.set_xlabel("SHAP value (impact on model output)", color="white", fontsize=11)
        ax2.tick_params(axis='x', colors='white', labelsize=10)
        ax2.tick_params(axis='y', colors='white', labelsize=10)
        
        cbar = plt.gcf().axes[-1]
        cbar.tick_params(colors='white', labelsize=9)
        cbar.set_ylabel(cbar.get_ylabel(), color='white', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig2, width='stretch')

    with st.expander("How to Read These Plots"):
        st.markdown(f"""
        - **Bar Chart:** Ranks features by their average impact on the model's predictions.
        - **Beeswarm Plot:** Shows the distribution of impacts for each feature. Each dot is a prediction.
            - **Position:** Right of zero means it pushed the price prediction higher; left means lower.
            - **Color:** Red dots are high feature values; blue dots are low feature values.
        
        *Analysis based on {len(X_sample):,} representative product samples.*
        """)

st.divider()

# Local Explanations
with st.container(border=True):
    st.subheader("🔬 Local Prediction Explanations")
    st.markdown("Let's break down a **single prediction**. Select a product instance below to see how the model arrived at its forecast.")

    sample_indices = X_sample.index.tolist()
    selected_idx = st.selectbox(
        "Select a product instance to analyze:",
        options=range(len(sample_indices)),
        format_func=lambda x: f"Product Sample #{x+1} (Index: {sample_indices[x]})",
        help="Each sample represents a unique product on a specific day."
    )
    
    random_index = sample_indices[selected_idx]

    if random_index is not None:
        instance = X_sample.loc[[random_index]]
        shap_value_instance = shap_values_sample[selected_idx:selected_idx+1]
        prediction = model.predict(instance)[0]

        st.markdown(f"##### Explaining Prediction for Sample #{selected_idx+1}")
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Model's Price Prediction", value=f"£{prediction:.2f}")
        col2.metric(label="Base Value (Average)", value=f"£{base_value:.2f}")
        col3.metric(label="Prediction Difference", value=f"£{prediction - base_value:+.2f}")

        st.markdown("##### Force Plot Breakdown")
        st.markdown("This visualization shows how each feature pushed the prediction up (red) or down (blue) from the baseline.")

        try:
            force_plot = shap.force_plot(
                base_value,
                shap_value_instance,
                instance,
                show=False,
                matplotlib=False
            )

            shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
            components.html(shap_html, height=200)
        except Exception as e:
            st.warning(f"Could not render interactive force plot: {e}")
            st.markdown("Showing top influential features instead:")
            
            feature_contributions = pd.DataFrame({
                'Feature': X_sample.columns,
                'Value': instance.values[0],
                'SHAP Impact': shap_value_instance[0]
            }).sort_values('SHAP Impact', key=abs, ascending=False).head(10)
            
            st.dataframe(feature_contributions.style.format({
                'Value': '{:.3f}',
                'SHAP Impact': '{:+.3f}'
            }), width='stretch')

        with st.expander("How to Read This Force Plot"):
            st.markdown(f"""
            This plot shows the forces pushing the prediction for this single instance.
            - The **base value (£{base_value:.2f})** is the average prediction over the dataset.
            - **<span style='color: #ff0d57;'>Red arrows</span>** show features that pushed the prediction **higher**.
            - **<span style='color: #008bfb;'>Blue arrows</span>** show features that pushed the prediction **lower**.
            - The final **bold** number is the model's output for this specific instance.
            
            The length of each arrow represents the magnitude of that feature's impact on the prediction.
            """, unsafe_allow_html=True)

# Optional: Feature Importance Table
with st.expander("📊 View Detailed Feature Importance Table"):
    feature_importance = pd.DataFrame({
        'Feature': X_sample.columns,
        'Mean |SHAP|': abs(shap_values_sample).mean(axis=0)
    }).sort_values('Mean |SHAP|', ascending=False)
    
    st.dataframe(
        feature_importance.style.format({'Mean |SHAP|': '{:.4f}'}).background_gradient(
            subset=['Mean |SHAP|'], cmap='YlOrRd'
        ),
        width='stretch',
        height=400
    )