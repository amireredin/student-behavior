import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Data Explorer", page_icon="🔍", layout="wide")

@st.cache_data
def load_data():
    data = pd.read_csv('xAPI-Edu-Data.csv')
    return data

def main():
    st.title("🔍 Data Explorer")

    data = load_data()

    st.header("Dataset Overview")
    st.write(data.head())
    
    st.subheader("Dataset Information")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.subheader("Statistical Summary")
    st.write(data.describe())

    st.header("Interactive Data Table")
    st.write("Explore the full dataset with sorting, filtering, and pagination capabilities:")
    st.dataframe(data, height=400)

    st.header("Feature Distribution Visualizations")

    # Feature selection
    feature = st.selectbox("Select a feature to analyze:", data.columns, key="feature_selectbox")
    plot_type = st.radio("Select plot type:", ('Histogram', 'Box Plot'), key="plot_type_radio")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Distribution of {feature}")
        if plot_type == 'Histogram':
            fig = px.histogram(data, x=feature, nbins=30, marginal="box")
        else:
            fig = px.box(data, y=feature)
        st.plotly_chart(fig)

    with col2:
        st.subheader(f"{feature} vs. Performance")
        if data[feature].dtype == 'object':
            fig = px.histogram(data, x='Class', color=feature, 
                               barnorm='percent', text_auto='.2f',
                               title=f"Performance Distribution by {feature}")
            fig.update_layout(bargap=0.1)
        else:
            fig = px.box(data, x='Class', y=feature, 
                         color='Class', notched=True,
                         title=f"{feature} Distribution Across Performance Levels")
        st.plotly_chart(fig)

    st.header("Correlation Matrix Heatmap")
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    corr_matrix = data[numeric_cols].corr()

    fig = px.imshow(corr_matrix, 
                    labels=dict(color="Correlation"),
                    x=numeric_cols,
                    y=numeric_cols,
                    color_continuous_scale='RdBu_r',
                    aspect="auto")
    fig.update_layout(title="Correlation Heatmap of Numeric Features")
    st.plotly_chart(fig)

    st.subheader("Top Correlations")
    corr_pairs = corr_matrix.unstack().sort_values(kind="quicksort", ascending=False).drop_duplicates()
    top_corr = corr_pairs[(corr_pairs != 1.0) & (abs(corr_pairs) > 0.5)]
    st.write(top_corr)

    st.header("Feature Relationship Explorer")
    
    st.write("Select features for scatter plot to explore their relationships:")
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Select X-axis", options=data.columns, key="x_axis_selectbox")
    with col2:
        y_axis = st.selectbox("Select Y-axis", options=data.columns, key="y_axis_selectbox")

    color_by = st.selectbox("Color by", options=['None'] + list(data.columns), key="color_by_selectbox")
    
    if color_by == 'None':
        fig = px.scatter(data, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
    else:
        fig = px.scatter(data, x=x_axis, y=y_axis, color=color_by, title=f"{y_axis} vs {x_axis}, colored by {color_by}")
    
    st.plotly_chart(fig)

    st.header("Data Distribution Across Performance Levels")
    
    categorical_cols = data.select_dtypes(include=['object']).columns
    selected_cat_feature = st.selectbox("Select a categorical feature:", categorical_cols, key="selected_cat_feature_selectbox")

    fig = px.sunburst(data, path=['Class', selected_cat_feature], values='raisedhands',
                      title=f"Distribution of {selected_cat_feature} Across Performance Levels")
    st.plotly_chart(fig)

    st.header("Data Quality Assessment")
    
    missing_data = data.isnull().sum()
    if missing_data.sum() > 0:
        st.subheader("Missing Data")
        st.write(missing_data[missing_data > 0])
    else:
        st.write("No missing data found in the dataset.")

    st.subheader("Outlier Detection")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    selected_num_feature = st.selectbox("Select a numeric feature for outlier detection:", numeric_cols, key="selected_num_feature_selectbox")

    fig = go.Figure()
    fig.add_trace(go.Box(y=data[selected_num_feature], name=selected_num_feature))
    fig.update_layout(title=f"Box Plot for {selected_num_feature}")
    st.plotly_chart(fig)

    Q1 = data[selected_num_feature].quantile(0.25)
    Q3 = data[selected_num_feature].quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data[selected_num_feature] < (Q1 - 1.5 * IQR)) | (data[selected_num_feature] > (Q3 + 1.5 * IQR))]
    st.write(f"Number of outliers detected: {len(outliers)}")

if __name__ == "__main__":
    main()
