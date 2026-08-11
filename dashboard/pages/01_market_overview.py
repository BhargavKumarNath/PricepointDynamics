import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_canonical_data
from utils import set_plot_style


# Page Configuration
st.set_page_config(page_title="Market Overview", layout="wide")

# Header
st.markdown("<h1 style='text-align: center; color: white;'>📈 Market Overview</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>A 30,000-foot view of the UK supermarket landscape, exploring each retailer's pricing strategy, product portfolio, and category focus.</p>", unsafe_allow_html=True)
st.divider()

#  Data Loading
df = load_canonical_data()

# Section 1: Pricing and Portfolio Analysis
st.subheader("At a Glance: Pricing & Portfolio")
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        #  1. Price Distribution Analysis
        st.markdown("##### Price Distribution by Supermarket")
        
        show_outliers = st.checkbox("Show outliers (extreme prices)", value=False, key="dist_outliers")
        
        set_plot_style()
        fig, ax = plt.subplots(figsize=(8, 5))
        
        sns.boxplot(
            x='supermarket', y='prices', data=df, 
            showfliers=show_outliers, palette='viridis', ax=ax,
            hue='supermarket', legend=False
        )
        
        ax.set_title('Product Price Distribution', fontsize=12)
        ax.set_ylabel('Price (£)')
        ax.set_xlabel('') 
        st.pyplot(fig, width='stretch')
        st.markdown("""
        **Insight:** This reveals the market's two-tiered structure. **Aldi** operates in a significantly lower price bracket, confirming its hard-discounter model, while the "Big Four" compete in a similar, higher price range.
        """)

with col2:
    with st.container(border=True):
        # 2. Product Portfolio Analysis
        st.markdown("##### Product Portfolio Size")
        
        portfolio_size = df.groupby('supermarket', observed=True)['canonical_name'].nunique().sort_values(ascending=False)
        
        set_plot_style()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=portfolio_size.index, y=portfolio_size.values, palette='mako', ax=ax, hue=portfolio_size.index, legend=False)
        
        ax.set_title('Number of Unique Products by Retailer', fontsize=12)
        ax.set_ylabel('Count of Unique Products')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig, width='stretch')
        st.markdown("""
        **Insight:** Sainsbury's, ASDA, and Tesco offer a vast range, positioning themselves as "one-stop-shops". **Aldi's** curated selection highlights a strategy focused on operational efficiency over choice.
        """)

st.divider()

# Section 2: Own Brand Strategy Analysis
st.subheader("Deep Dive: Own Brand Strategy")
col3, col4 = st.columns([1, 2]) 

with col3:
    with st.container(border=True, height=450): 
        # 3a. Own Brand Percentage
        st.markdown("##### Percentage of Own Brand")
        own_brand_percentage = df.groupby('supermarket', observed=True)['own_brand'].mean() * 100
        st.dataframe(own_brand_percentage.sort_values(ascending=False).map("{:.2f}%".format), width='stretch')
        st.markdown("""
        **Insight:** Contrary to common perception, it's the larger supermarkets like **ASDA** that have the highest proportion of own-brand items, showing their reliance on these lines to compete.
        """)

with col4:
    with st.container(border=True, height=450): 
        # 3b. Own Brand vs Branded Count
        st.markdown("##### Product Listings: Own Brand vs. Branded")
        
        set_plot_style()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=df, x='supermarket', hue='own_brand', palette={True: '#6495ED', False: '#FF7F50'}, ax=ax)
        
        ax.set_title('') 
        ax.set_ylabel('Number of Listings (Log Scale)')
        ax.set_xlabel('Supermarket')
        ax.set_yscale('log')
        legend = plt.legend(title='Is Own Brand?', labels=['Branded', 'Own Brand'])
        plt.setp(legend.get_texts(), color='white') 
        plt.setp(legend.get_title(), color='white') 
        
        st.pyplot(fig, width='stretch')