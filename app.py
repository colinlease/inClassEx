"""
Simple Exploratory Data Analysis (EDA) Streamlit App
Loads the Iris dataset and provides interactive visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Page configuration
st.set_page_config(
    page_title="Iris Dataset EDA",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("📊 Iris Dataset Exploratory Data Analysis")
st.markdown("Interactive tool to explore the famous Iris dataset")

# Load the iris dataset
iris_data = load_iris()
iris_dataframe = pd.DataFrame(
    data=iris_data.data,
    columns=iris_data.feature_names
)
iris_dataframe['species'] = iris_data.target_names[iris_data.target]

# Display basic information
st.header("Dataset Overview")

# Show first rows
st.subheader("First 5 Rows of Data")
st.dataframe(iris_dataframe.head())

# Show dataset shape
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Rows", iris_dataframe.shape[0])
with col2:
    st.metric("Total Columns", iris_dataframe.shape[1])
with col3:
    st.metric("Unique Species", iris_dataframe['species'].nunique())

# Display summary statistics
st.subheader("Summary Statistics")
st.dataframe(iris_dataframe.describe())

# Interactive visualization section
st.header("Interactive Visualizations")

# Get numeric columns (exclude the species column)
numeric_columns = iris_dataframe.select_dtypes(include=[np.number]).columns.tolist()

# Create two columns for user selections
col1, col2 = st.columns(2)

with col1:
    # Histogram selection
    st.subheader("Histogram")
    histogram_column = st.selectbox(
        "Select a numeric column for histogram:",
        numeric_columns,
        key="histogram_select"
    )
    
    # Number of bins slider
    num_bins = st.slider(
        "Number of bins:",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        key="bins_slider"
    )
    
    # Create and display histogram
    fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
    ax_hist.hist(iris_dataframe[histogram_column], bins=num_bins, color='steelblue', edgecolor='black')
    ax_hist.set_xlabel(histogram_column)
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title(f"Distribution of {histogram_column}")
    ax_hist.grid(axis='y', alpha=0.3)
    st.pyplot(fig_hist)

with col2:
    # Scatter plot selection
    st.subheader("Scatter Plot")
    scatter_col1 = st.selectbox(
        "Select X-axis column:",
        numeric_columns,
        key="scatter_x"
    )
    scatter_col2 = st.selectbox(
        "Select Y-axis column:",
        numeric_columns,
        index=1 if len(numeric_columns) > 1 else 0,
        key="scatter_y"
    )
    
    # Create and display scatter plot
    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
    
    # Color code by species
    species_colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
    for species in iris_dataframe['species'].unique():
        mask = iris_dataframe['species'] == species
        ax_scatter.scatter(
            iris_dataframe.loc[mask, scatter_col1],
            iris_dataframe.loc[mask, scatter_col2],
            label=species,
            alpha=0.6,
            s=80,
            color=species_colors.get(species, 'gray')
        )
    
    ax_scatter.set_xlabel(scatter_col1)
    ax_scatter.set_ylabel(scatter_col2)
    ax_scatter.set_title(f"{scatter_col1} vs {scatter_col2}")
    ax_scatter.legend()
    ax_scatter.grid(alpha=0.3)
    st.pyplot(fig_scatter)

# Data download section
st.header("Download Data")
st.markdown("Download the dataset as CSV:")

# Convert dataframe to CSV
csv_data = iris_dataframe.to_csv(index=False)
st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name="iris_dataset.csv",
    mime="text/csv"
)
