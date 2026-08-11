import streamlit as st

st.set_page_config(
    page_title="PricePoint Dynamics",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialise session state for cross-page data persistence
if "data_loaded" not in st.session_state:
    st.session_state["data_loaded"] = False

# Header
st.title("📝 PricePoint Dynamics")
st.subheader("An Interactive Dashboard for the UK Supermarket Competitive Landscape")

st.markdown("""
This dashboard presents the findings from an end-to-end data science project
analysing over 9.5 million daily price records from 5 major UK supermarkets.
The goal is to uncover deep competitive structure, model market dynamics,
and predict future prices using machine learning.
""")
st.markdown("---")

st.header("🏆 Project Highlights")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Price Records Analysed", value="9.5 Million")
with col2:
    st.metric(label="Comparable Products Identified", value="67,000+")
with col3:
    st.metric(label="Price Prediction Model MAE", value="£0.14")
st.markdown("---")

# Instruction
st.header("Navigating the Dashboard")
st.info(""" 
Use the sidebar on the left to navigate through the different phases of the analysis:
        
- **📈 Market Overview:** Get a high-level view of retailer pricing and product portfolios.
- **🛒 Basket Analysis:** Compare the cost of standardised shopping baskets across stores.
- **🤖 Price Predictor:** An interactive tool to predict product prices using our ML model.
- **🧠 Model Insights:** Understand *why* our model makes its predictions using SHAP.
- **🌐 Market Dynamics:** Explore price leadership and market competitiveness over time.

""")

st.sidebar.success("Select a page above to begin your analysis")
