import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_agraph import agraph, Node, Edge, Config
from data_loader import load_market_dispersion, load_price_leadership
from utils import set_plot_style

# Page Configuration
st.set_page_config(page_title="Market Dynamics", layout="wide")

st.markdown("""
<style>
    /* Main metric value */
    div[data-testid="stMetricValue"] {
        color: #FFFFFF;
    }
    /* Delta indicator (e.g., "-2.0 days") */
    div[data-testid="stMetricDelta"] {
        color: #B0B0B0; /* A neutral gray */
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: white;'>🌐 Market Dynamics & Price Leadership</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>This section explores the strategic interactions between retailers over time, answering the question: <b>Who leads, and who follows?</b></p>", unsafe_allow_html=True)
st.divider()

# Load pre-computed data
with st.spinner("Loading market dynamics data..."):
    market_dispersion = load_market_dispersion()
    leader_df = load_price_leadership()

# Check if data loaded successfully
if market_dispersion is None or leader_df is None:
    st.error("""
    ### Pre-computed market dynamics data not available
    
    To enable market dynamics analysis, run:
    ```bash
    python run.py precompute
    ```
    """)
    st.stop()

st.success("✓ Market dynamics data loaded successfully!")

tab1, tab2 = st.tabs(["📊 Market Competitiveness", "🕸️ Price Leadership Network"])

with tab1:
    st.header("Market Competitiveness Over Time")
    st.markdown("This chart measures **price dispersion** (the average variation of prices for the same product across stores). A lower value indicates a more competitive market.")
    
    latest_dispersion = market_dispersion.iloc[-1]
    avg_dispersion = market_dispersion.mean()
    st.metric(
        label="Latest Market Dispersion",
        value=f"{latest_dispersion:.3f}",
        delta=f"{(latest_dispersion - avg_dispersion):.3f} vs. avg",
        help="This is the average coefficient of variation for prices on the last available day."
    )

    with st.container(border=True):
        set_plot_style()
        fig, ax = plt.subplots(figsize=(12, 6))
        market_dispersion.plot(ax=ax, label='Mean Daily Dispersion', color='#4B8BFF', linewidth=1.5, alpha=0.9)
        market_dispersion.rolling(7).mean().plot(ax=ax, linestyle='--', color='#FFAF4B', label='7-Day Rolling Average', linewidth=2.5)
        ax.set_title("Market-Wide Price Dispersion", fontsize=16)
        ax.set_ylabel("Avg. Coefficient of Variation")
        ax.set_xlabel("Date")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()
        st.pyplot(fig, width='stretch')

    st.info("**Insight:** After a volatile period in mid-January, the market settled into a stable equilibrium. The level of price difference between retailers is not escalating into a price war, nor is it diminishing.", icon="💡")

with tab2:
    st.header("Who Leads and Who Follows?")
    st.markdown("This network visualizes price leadership. An arrow from `A` to `B` means `A`'s price changes tend to happen before `B`'s.")

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        nodes, edges = [], []
        all_supermarkets_list = ['Aldi', 'ASDA', 'Morrisons', 'Sains', 'Tesco']
        node_colors = {'Aldi': '#FF4B4B', 'ASDA': '#3DDC97', 'Morrisons': '#FFAF4B', 'Sains': '#966DFF', 'Tesco': '#4B8BFF'}

        for retailer in all_supermarkets_list:
            nodes.append(Node(id=retailer, label=retailer, size=25, color=node_colors.get(retailer), font={'color': 'white', 'size': 18}))

        for _, row in leader_df.iterrows():
            lag_label = f"{abs(row['median_lag_days']):.0f} days"
            source, target = (row['leader'], row['follower']) if row['median_lag_days'] > 0 else (row['follower'], row['leader'])
            edges.append(Edge(source=source, target=target, label=lag_label, color='#808080', font={'color': '#B0B0B0', 'size': 14}, arrows='to'))
        
        config = Config(
            width='100%', 
            height=550, 
            directed=True, 
            physics={
                "forceAtlas2Based": {
                    "gravitationalConstant": -50, 
                    "centralGravity": 0.01, 
                    "springLength": 230, 
                    "springConstant": 0.08
                }, 
                "minVelocity": 0.75, 
                "solver": "forceAtlas2Based"
            }, 
            interaction={'navigationButtons': True, 'tooltipDelay': 200}, 
            nodeHighlightBehavior=True
        )
        
        with st.container(border=True):
            agraph(nodes=nodes, edges=edges, config=config)

    with col2:
        st.subheader("Leadership Data")
        
        if not leader_df.empty:
            top_leader = leader_df['leader'].mode()[0]
            fastest_follower_row = leader_df.loc[leader_df['median_lag_days'].abs().idxmin()]
            
            st.metric(label="Primary Mover", value=top_leader, help="The retailer that most frequently leads price changes.")
            
            st.metric(
                label="Fastest Follower",
                value=fastest_follower_row['follower'],
                delta=f"{fastest_follower_row['median_lag_days']:.1f} days",
                help=f"This retailer reacts the quickest, following price changes from {fastest_follower_row['leader']}."
            )
            
            st.markdown(f"*Based on {leader_df['n_products_analyzed'].sum():,} product comparisons*")
        
        st.dataframe(
            leader_df[['leader', 'follower', 'median_lag_days', 'n_products_analyzed']].sort_values('median_lag_days', ascending=False, key=abs), 
            width='stretch'
        )

    st.info("**Insight:** The graph and table clearly show that Aldi is a primary price-setter. Arrows consistently point from Aldi to the 'Big Four,' with a lag of several days. Among the 'Big Four,' the relationships are much faster and more reciprocal, indicating a tight, reactive competitive cluster.", icon="💡")