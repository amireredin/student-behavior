import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="Predictor", page_icon="⌨️", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv('xAPI-Edu-Data.csv')
    return data

@st.cache_data
def preprocess_data(data, scale_features=False, is_input=False):
    label_enc = LabelEncoder()
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].fillna('Unknown')
            data[col] = label_enc.fit_transform(data[col])
    
    if not is_input and 'Class' in data.columns:
        feature_cols = data.drop('Class', axis=1).columns
    else:
        feature_cols = data.columns

    if scale_features:
        scaler = StandardScaler()
        data[feature_cols] = scaler.fit_transform(data[feature_cols])
    
    return data

# Train Random Forest model
@st.cache_resource
def train_random_forest(data):
    X = data.drop('Class', axis=1)
    y = data['Class']
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

# Train SVM model
@st.cache_resource
def train_svm(data):
    X = data.drop('Class', axis=1)
    y = data['Class']
    model = SVC(probability=True, random_state=42)
    model.fit(X, y)
    return model

# Train Logistic Regression model
@st.cache_resource
def train_logistic_regression(data):
    X = data.drop('Class', axis=1)
    y = data['Class']
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    return model

# Feature input form
def user_input_features():
    st.header('Input Features')
    data = load_data()
    features = data.drop('Class', axis=1).columns
    input_data = {}

    for feature in features:
        if data[feature].dtype == 'object':
            input_data[feature] = st.selectbox(f'Select {feature}', data[feature].unique())
        elif feature in ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']:
            input_data[feature] = st.number_input(f'Enter {feature}', value=int(data[feature].mean()), step=1, format="%d")
        else:
            input_data[feature] = st.number_input(f'Enter {feature}', value=float(data[feature].mean()))

    input_df = pd.DataFrame(input_data, index=[0])
    return input_df

# Plot feature importance
def plot_feature_importance(model, X, n=10):
    if isinstance(model, RandomForestClassifier):
        feature_importances = model.feature_importances_
        indices = np.argsort(feature_importances)[-n:]
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.barh(range(len(indices)), feature_importances[indices], color="b", align="center")
        plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
        plt.xlabel("Relative Importance")
        st.pyplot(plt)
    else:
        st.write("Feature importances are not available for this model type.")

# Define a dictionary for predicted performance levels
performance_levels = {
    0: 'High',
    1: 'Low',
    2: 'Medium'
}

# Main function
def main():
    
    st.title("🎓 Student Performance Predictor")

    # Load and preprocess data
    data = load_data()

    # Model selection
    model_choice = st.selectbox(
        "Choose a model for prediction",
        ('Random Forest', 'Support Vector Machine', 'Logistic Regression')
    )

    # Check if the selected model requires scaling
    scale_features = model_choice in ['Support Vector Machine', 'Logistic Regression']
    
    # Preprocess data with scaling if needed
    preprocessed_data = preprocess_data(data, scale_features=scale_features)

    # Train model based on user selection
    if model_choice == 'Random Forest':
        model = train_random_forest(preprocessed_data)
    elif model_choice == 'Support Vector Machine':
        model = train_svm(preprocessed_data)
    elif model_choice == 'Logistic Regression':
        model = train_logistic_regression(preprocessed_data)
    else:
        st.error("Invalid model choice. Please select a valid model.")
        return

    # Get user input features
    input_df = user_input_features()

    # Display user input features
    st.header("Input Features")
    st.write(input_df)

    # Preprocess input data with scaling if needed
    input_data = preprocess_data(input_df, scale_features=scale_features, is_input=True)

    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.header("Prediction Output")
    st.write(f"**Predicted Performance Level:** {performance_levels[prediction[0]]}")
    st.write(f"**Confidence Score:** {np.max(prediction_proba) * 100:.2f}%")

    # Feature importance
    if model_choice == 'Random Forest':
        st.header("Feature Impact")
        num_features = st.slider("Select the number of top features to display:", min_value=1, max_value=len(preprocessed_data.columns) - 1, value=10)
        plot_feature_importance(model, preprocessed_data.drop('Class', axis=1), n=num_features)

if __name__ == "__main__":
    main()