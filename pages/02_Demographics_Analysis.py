import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Demographic Analysis", page_icon="🌍", layout="wide")

@st.cache_data
def load_data():
    data = pd.read_csv('xAPI-Edu-Data.csv')
    return data

def main():
    st.title("🌍 Demographic Analysis")

    data = load_data()

    st.header("Overview")
    st.markdown("""
    This section provides insights into how various demographic factors such as nationality and gender influence student academic performance.
    """)

    st.header("Gender Distribution")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Pie Chart of Gender Distribution")
        gender_dist = data['gender'].value_counts()
        fig_gender_dist = px.pie(gender_dist, values=gender_dist.values, names=gender_dist.index,
                                 title="Gender Distribution", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_gender_dist)

    with col2:
        st.subheader("Performance Level Breakdown by Gender")
        fig_perf_gender = px.histogram(data, x='Class', color='gender', barmode='group',
                                       title="Performance Distribution by Gender",
                                       category_orders={"Class": ["L", "M", "H"]},
                                       labels={'Class': 'Performance Level', 'gender': 'Gender'})
        st.plotly_chart(fig_perf_gender)

    st.header("Nationality Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("World Map Highlighting Student Origins")
        nationality_dist = data['NationalITy'].value_counts().reset_index()
        nationality_dist.columns = ['Nationality', 'Count']
        country_code_mapping = {
            'KW': 'Kuwait'
        }
        nationality_dist['Nationality'] = nationality_dist['Nationality'].replace(country_code_mapping)
        fig_nationality_map = px.choropleth(nationality_dist, locations="Nationality", locationmode='country names', color="Count",
                                            title="Student Origins by Nationality", color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig_nationality_map)

    with col2:
        st.subheader("Bar Chart of Performance by Nationality")
        fig_perf_nationality = px.histogram(data, x='Class', color='NationalITy', barmode='group',
                                            title="Performance Distribution by Nationality",
                                            category_orders={"Class": ["L", "M", "H"]},
                                            labels={'Class': 'Performance Level', 'NationalITy': 'Nationality'})
        st.plotly_chart(fig_perf_nationality)

    st.header("Educational Stage Analysis")
    st.subheader("Stacked Bar Chart of Performance by Educational Stage")
    fig_perf_stage = px.histogram(data, x='StageID', color='Class', barmode='stack',
                                  title="Performance Distribution by Educational Stage",
                                  category_orders={"Class": ["L", "M", "H"]},
                                  labels={'Class': 'Performance Level', 'StageID': 'Educational Stage'})
    st.plotly_chart(fig_perf_stage)

    
if __name__ == "__main__":
    main()
