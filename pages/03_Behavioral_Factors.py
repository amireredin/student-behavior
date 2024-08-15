import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Behavioral Factors Analysis", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    data = pd.read_csv('xAPI-Edu-Data.csv')
    return data

def main():
    st.title("📊 Behavioral Factors Analysis")

    data = load_data()

    st.header("Classroom Engagement")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Scatter Plot: Raised Hands vs. Performance")
        fig_raised_perf = px.scatter(data, x='raisedhands', y='Class', color='Class',
                                     title="Raised Hands vs. Performance",
                                     labels={'raisedhands': 'Raised Hands', 'Class': 'Performance Level'},
                                     category_orders={"Class": ["L", "M", "H"]})
        st.plotly_chart(fig_raised_perf)

    with col2:
        st.subheader("Histogram: Distribution of Raised Hands")
        fig_raised_dist = px.histogram(data, x='raisedhands', nbins=30, title="Distribution of Raised Hands",
                                       labels={'raisedhands': 'Raised Hands'})
        st.plotly_chart(fig_raised_dist)

    st.header("Resource Utilization")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Bar Chart: Average Resource Visits by Performance Level")
        avg_resources = data.groupby('Class')['VisITedResources'].mean().reset_index()
        fig_avg_resources = px.bar(avg_resources, x='Class', y='VisITedResources', color='Class',
                                   title="Average Resource Visits by Performance Level",
                                   labels={'Class': 'Performance Level', 'VisITedResources': 'Average Resource Visits'},
                                   category_orders={"Class": ["L", "M", "H"]})
        st.plotly_chart(fig_avg_resources)

    with col2:
        st.subheader("Scatter Plot: Resource Visits vs. Discussion Group Participation")
        fig_resources_discussion = px.scatter(data, x='VisITedResources', y='Discussion', color='Class',
                                              title="Resource Visits vs. Discussion Group Participation",
                                              labels={'VisITedResources': 'Resource Visits', 'Discussion': 'Discussion Group Participation'},
                                              category_orders={"Class": ["L", "M", "H"]})
        st.plotly_chart(fig_resources_discussion)

    st.header("Absence Patterns")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Pie Chart: Absence Days (Above-7 vs. Under-7)")
        absence_dist = data['StudentAbsenceDays'].value_counts()
        fig_absence_dist = px.pie(absence_dist, values=absence_dist.values, names=absence_dist.index,
                                  title="Absence Days (Above-7 vs. Under-7)",
                                  labels={'index': 'Absence Days', 'value': 'Count'},
                                  color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_absence_dist)

    with col2:
        st.subheader("Performance Comparison Between Absence Groups")
        fig_absence_perf = px.histogram(data, x='Class', color='StudentAbsenceDays', barmode='group',
                                        title="Performance Comparison Between Absence Groups",
                                        category_orders={"Class": ["L", "M", "H"]},
                                        labels={'Class': 'Performance Level', 'StudentAbsenceDays': 'Absence Days'})
        st.plotly_chart(fig_absence_perf)

if __name__ == "__main__":
    main()
