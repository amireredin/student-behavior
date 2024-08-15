import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Parent Involvement Impact", page_icon="👪", layout="wide")

@st.cache_data
def load_data():
    data = pd.read_csv('xAPI-Edu-Data.csv')
    return data

def main():
    st.title("👪 Parent Involvement Impact")

    data = load_data()

    st.header("Survey Participation Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Performance by Parent Survey Participation")
        fig_survey_participation = px.histogram(data, x='Class', color='ParentschoolSatisfaction', barmode='stack',
                                                title="Performance by Parent Survey Participation",
                                                category_orders={"Class": ["L", "M", "H"]},
                                                labels={'Class': 'Performance Level', 'ParentschoolSatisfaction': 'Parent Survey Participation'})
        st.plotly_chart(fig_survey_participation)

    st.header("School Satisfaction Effect")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Performance by Parent Satisfaction")
        fig_satisfaction = px.histogram(data, x='Class', color='ParentschoolSatisfaction', barmode='group',
                                        title="Performance by Parent Satisfaction",
                                        category_orders={"Class": ["L", "M", "H"], "ParentschoolSatisfaction": ["Bad", "Good"]},
                                        labels={'Class': 'Performance Level', 'ParentschoolSatisfaction': 'Parent Satisfaction'})
        st.plotly_chart(fig_satisfaction)

    st.header("Parent Responsibility Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution of Responsible Parent (Mother vs. Father)")
        responsible_parent_dist = data['Relation'].value_counts()
        fig_responsible_parent = px.pie(responsible_parent_dist, values=responsible_parent_dist.values, names=responsible_parent_dist.index,
                                        title="Distribution of Responsible Parent (Mother vs. Father)",
                                        labels={'index': 'Responsible Parent', 'value': 'Count'},
                                        color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_responsible_parent)

    with col2:
        st.subheader("Performance Comparison Based on Responsible Parent")
        fig_perf_responsible_parent = px.histogram(data, x='Class', color='Relation', barmode='group',
                                                   title="Performance Comparison Based on Responsible Parent",
                                                   category_orders={"Class": ["L", "M", "H"]},
                                                   labels={'Class': 'Performance Level', 'Relation': 'Responsible Parent'})
        st.plotly_chart(fig_perf_responsible_parent)

if __name__ == "__main__":
    main()
