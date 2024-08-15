import streamlit as st

st.set_page_config(page_title="About", page_icon="❔", layout="wide")

def main():

    st.title("❔ About the Project")

    st.header("Project Methodology")
    st.markdown("""
    Our project focuses on predicting and analyzing student academic performance through a structured approach involving data collection, exploration, and modeling. The methodology includes the following steps:

1. **Data Collection and Preprocessing**:
   - Utilized the xAPI-Edu-Data dataset, which contains features related to student demographics, academic behavior, and parental involvement.
   - Performed data cleaning by handling missing values and encoding categorical variables using Label Encoding to convert them into numerical format.
   - Applied feature scaling using StandardScaler for algorithms like SVM and Logistic Regression to ensure consistent performance.

2. **Exploratory Data Analysis (EDA)**:
   - Conducted a thorough analysis to identify patterns and correlations within the dataset.
   - Visualized the impact of demographic factors such as nationality, gender, and grade level on student performance.
   - Examined behavioral factors including participation in class and resource usage to understand their influence on academic outcomes.

3. **Model Development**:
   - Developed and trained machine learning models: Random Forest, Support Vector Machine, and Logistic Regression.
   - Assessed model performance using metrics such as accuracy and feature importance, selecting the Random Forest model for its interpretability and robustness.

4. **Prediction and Analysis**:
   - Created a prediction tool that allows for the estimation of student performance based on input features.
   - Analyzed feature importance to determine the most influential factors in predicting performance, providing valuable insights for educational stakeholders.

5. **Parent Involvement Analysis**:
   - Investigated the significance of parental involvement in student success through survey data.
   - Evaluated the relationship between parental satisfaction and student performance, highlighting the critical role of family engagement.

    """)

    st.header("Findings")
    st.markdown("""This project leverages insights from prior analyses of the xAPI-Edu-Data dataset, revealing key factors influencing student performance:

1. **Subject Performance**:
   - No students failed Geology, while IT, Chemistry, and Math had higher failure rates. This may be related to higher participation and attendance in Geology classes.
   - **Reference**: Bar plot showing subject-wise failure rates.

2. **Parental Satisfaction**:
   - Students with satisfied parents generally performed better. Those with dissatisfied parents tended to perform worse.
   - Students whose mothers were responsible for them showed higher chances of performing well.
   - **Reference**: Boxplot of student performance vs. parental satisfaction.

3. **Class Participation**:
   - High-performing students were more active in class, while low performers participated less.
   - The lowest performers rarely accessed course resources, while the most consistent habits were observed in both high and low performers.
   - **Reference**: Plot of class participation and resource usage.

4. **Gender and Demographics**:
   - On average, female students outperformed male students, although gender alone doesn't fully explain performance variations.
   - Jordanian students showed above-average performance, indicating a positive impact on certain student groups.
   - **Reference**: Bar plot comparing performance by gender and nationality.

5. **Behavioral Factors**:
   - Students who frequently raised hands, visited resources, and viewed announcements generally achieved higher performance levels.
   - Some students, despite high engagement, still received low grades, indicating other influencing factors.
   - **Reference**: Plot of engagement metrics vs. performance.

6. **Parent-Child Relationship**:
   - A strong relationship with mothers positively affected student success.
   - **Reference**: Boxplot of performance vs. parental involvement.

These findings illustrate the complexity of academic performance, emphasizing the roles of subject engagement, parental influence, and student behavior. By understanding these dynamics, educators can better support student success through targeted interventions.
""")

    st.header("Conclusion")
    st.markdown("""This project illustrates the power of data-driven approaches in understanding and predicting student performance. By employing advanced machine learning techniques, we identified key determinants of academic success, offering a predictive tool for educators and policymakers. The analysis emphasizes the multifaceted nature of student performance, influenced by demographic, behavioral, and parental factors. 
                These insights can inform targeted interventions and strategies to support students, ultimately contributing to improved educational outcomes. The findings underscore the importance of a holistic approach in addressing educational challenges, advocating for collaborative efforts between educators, parents, and policymakers.
    """)


    st.header("Data Source")
    st.markdown("""
    **Data Source**:
    The dataset used for this analysis is the [xAPI-Edu-Data dataset](https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data/data).
    """)

    st.header("Team Information")
    st.markdown("""
    This project was developed by a dedicated team of data enthusiasts. Our team members include:
    - **Amir Hossein Shahdadian**
    - **Kimia Asadzadeh**

    We collaborated to bring together our expertise in data science, machine learning, and software development to create this predictive model and deploy it as a user-friendly application.
    """)

if __name__ == "__main__":
    main()
