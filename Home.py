import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Student Performance Analysis", page_icon="📚", layout="wide")

@st.cache_data
def load_data():
    data = pd.read_csv('xAPI-Edu-Data.csv')
    return data

def main():
    st.title("📚 Student Academic Performance Analysis")

    st.markdown("""
    ## 🎯 Project Overview
    This application analyzes the academic performance of students based on various factors 
    including demographics, behavioral patterns, and parental involvement. Our goal is to 
    provide insights into what influences student success and offer predictive tools for 
    educators and administrators.
    """)

    data = load_data()

    st.header("📊 Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", len(data))
    with col2:
        st.metric("Features", data.shape[1])
    with col3:
        st.metric("Nationalities", data['NationalITy'].nunique())
    with col4:
        st.metric("Topics", data['Topic'].nunique())

    st.subheader("📈 Quick Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Male Students", len(data[data['gender'] == 'M']))
    with col2:
        st.metric("Female Students", len(data[data['gender'] == 'F']))
    with col3:
        st.metric("Educational Stages", data['StageID'].nunique())

    st.header("🔍 Key Insights")

    # Gender Distribution
    st.subheader("Gender Distribution")
    gender_dist = data['gender'].value_counts()
    fig_gender = px.pie(values=gender_dist.values, names=gender_dist.index, 
                        title="Gender Distribution", color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_gender)

    # Top Nationalities
    st.subheader("Top 5 Nationalities")
    top_nationalities = data['NationalITy'].value_counts().nlargest(5)
    fig_nationalities = px.bar(x=top_nationalities.index, y=top_nationalities.values, 
                               title="Top 5 Nationalities", color=top_nationalities.values,
                               color_continuous_scale=px.colors.sequential.Viridis)
    st.plotly_chart(fig_nationalities)

    # Performance Distribution
    st.subheader("Performance Level Distribution")
    perf_dist = data['Class'].value_counts().sort_index()
    fig_perf = px.bar(x=perf_dist.index, y=perf_dist.values, 
                      title="Performance Level Distribution", 
                      labels={'x': 'Performance Level', 'y': 'Count'},
                      color=perf_dist.values, color_continuous_scale=px.colors.sequential.Plasma)
    st.plotly_chart(fig_perf)

    # Absence Days vs Performance
    st.subheader("Absence Days vs Performance")
    fig_absence = px.bar(data, x='Class', color='StudentAbsenceDays', 
                         title="Absence Days vs Performance",
                         category_orders={"Class": ["L", "M", "H"]},
                         labels={'Class': 'Performance Level', 'StudentAbsenceDays': 'Absence Days'})
    st.plotly_chart(fig_absence)

    # Parent Involvement
    st.subheader("Parent Involvement")
    fig_parent = go.Figure()
    fig_parent.add_trace(go.Bar(x=['Survey Answered', 'Survey Not Answered'], 
                                y=[data['ParentAnsweringSurvey'].value_counts()['Yes'],
                                   data['ParentAnsweringSurvey'].value_counts()['No']],
                                name='Parent Survey', marker_color=['#72B7B2', '#F4A582']))
    fig_parent.add_trace(go.Bar(x=['Satisfied', 'Not Satisfied'], 
                                y=[data['ParentschoolSatisfaction'].value_counts()['Good'],
                                   data['ParentschoolSatisfaction'].value_counts()['Bad']],
                                name='School Satisfaction', marker_color=['#4DAF4A', '#E41A1C']))
    fig_parent.update_layout(title='Parent Involvement and Satisfaction', barmode='group')
    st.plotly_chart(fig_parent)

    st.header("🧭 Navigation Guide")
    st.markdown("""
    Explore the different aspects of our analysis using the sidebar navigation:
    
    1. **Data Explorer**: Dive deep into the dataset, explore individual features, and their distributions.
    2. **Demographics Analysis**: Understand how factors like nationality, gender, and grade level impact performance.
    3. **Behavioral Factors**: Analyze how student behavior, such as participation and resource usage, correlates with performance.
    4. **Parent Involvement Impact**: Discover the influence of parental involvement on student academic outcomes.
    5. **Modeling**: Explore predictive models that estimate student performance based on various factors.
    6. **Predictor**: Use our trained model to predict student performance by inputting various factors.
    7. **About**: Learn more about our project, its goals, and methodologies used.
                
    Feel free to navigate through these sections to gain comprehensive insights into student academic performance!
    """)

    st.header("📝 About the Dataset")
    st.markdown("""
    This dataset was collected from a Learning Management System (LMS) called Kalboard 360. It includes:
    - 480 student records
    - 17 features including demographic, academic, and behavioral data
    - Data collected over two semesters
    - Students from various nationalities and grade levels
    
    The target variable 'Class' represents the performance level of students:
    - Low-Level: 0 to 69
    - Middle-Level: 70 to 89
    - High-Level: 90 to 100
    
    For more details on specific features, please refer to the Data Explorer section.
    """)

if __name__ == "__main__": 
    main()