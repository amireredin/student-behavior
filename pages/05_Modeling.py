import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc, precision_recall_curve

st.set_page_config(page_title="Modeling", page_icon="📈", layout="wide")

# Cache data loading
@st.cache_data
def load_data():
    data = pd.read_csv('xAPI-Edu-Data.csv')
    return data

# Preprocess data
def preprocess_data(data, scale_features=False):
    # Handle categorical variables
    label_enc = LabelEncoder()
    class_mapping = {}  # To store the mapping of original classes to encoded values
    
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].fillna('Unknown')  # Fill missing values with 'Unknown'
            data[col] = label_enc.fit_transform(data[col])
            class_mapping[col] = dict(zip(label_enc.classes_, label_enc.transform(label_enc.classes_)))
    
    # Separate features and target
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    # Scale features if specified
    if scale_features:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X, y, class_mapping

# Plot feature importance
def plot_feature_importance(model, X, n=10):
    if isinstance(model, RandomForestClassifier):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-n:]
            features = X.columns[indices]
            fig, ax = plt.subplots()
            ax.barh(range(len(indices)), importances[indices], color='b', align='center')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels(features)
            ax.set_title('Top {} Feature Importances'.format(n))
            st.pyplot(fig)
        else:
            st.write("Selected model does not have feature importances attribute.")
    else:
        st.write("Feature importances are only available for Random Forest model.")

# Plot confusion matrix
def plot_conf_matrix(y_true, y_pred, labels):
    conf_matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    st.pyplot(fig)

# Plot ROC Curve for multiclass
def plot_multiclass_roc_curve(model, X_test, y_test, n_classes):
    y_score = model.predict_proba(X_test)
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_score[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig, ax = plt.subplots()
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} ROC curve (area = {roc_auc[i]:0.2f})')
    
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)

# Plot Precision-Recall Curve for multiclass
def plot_multiclass_precision_recall_curve(model, X_test, y_test, n_classes):
    y_score = model.predict_proba(X_test)
    precision = {}
    recall = {}
    pr_auc = {}

    fig, ax = plt.subplots()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test, y_score[:, i], pos_label=i)
        pr_auc[i] = auc(recall[i], precision[i])
        ax.plot(recall[i], precision[i], lw=2, label=f'Class {i} PR curve (area = {pr_auc[i]:0.2f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    st.pyplot(fig)

# Main function
def main():
    st.title("📈 Modeling Student Performance")

    # Load data
    data = load_data()
    
    # Display data overview
    st.header("Data Overview")
    st.write(data.head())

    # Preprocess data
    st.header("Preprocessing Data")
    X, y, class_mapping = preprocess_data(data)
    st.success("Preprocessing Complete")

    # Show class mapping
    st.subheader("Class Encoding Mapping")
    for col, mapping in class_mapping.items():
        st.write(f"{col}: {mapping}")

    # Split data
    st.header("Splitting Data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    st.write("Training set size:", X_train.shape)
    st.write("Test set size:", X_test.shape)

    # Model selection
    st.header("Model Selection")
    model_choice = st.selectbox(
        "Choose a model",
        ('Logistic Regression', 'Random Forest', 'Support Vector Machine')
    )

    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=200),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42)
    }
    
    model = models[model_choice]
    
    # Check if the selected model requires scaling
    scale_features = model_choice in ['Logistic Regression', 'Support Vector Machine']
    
    # Preprocess data with scaling based on model choice
    X_train, _, _ = preprocess_data(pd.concat([X_train, y_train], axis=1), scale_features=scale_features)
    X_test, _, _ = preprocess_data(pd.concat([X_test, y_test], axis=1), scale_features=scale_features)
    
    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Performance metrics
    acc = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    # Display performance metrics
    st.header(f"{model_choice} Results")
    st.write(f"**Accuracy:** {acc}")
    st.write(f"**Precision:** {precision}")
    st.write(f"**Recall:** {recall}")
    st.write(f"**F1-Score:** {f1_score}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    plot_conf_matrix(y_test, y_pred, labels=model.classes_)

    # Classification Report
    st.subheader("Classification Report")
    report_df = pd.DataFrame(class_report).transpose()
    exclude_rows = ['accuracy', 'macro avg', 'weighted avg']
    filtered_report_df = report_df[~report_df.index.isin(exclude_rows)]
    st.write(filtered_report_df)

    # Feature Importance (only available for Random Forest)
    if model_choice == 'Random Forest':
        st.subheader("Feature Importance")
        top_n = st.slider("Select top N features", min_value=1, max_value=16, value=10)
        plot_feature_importance(model, X, top_n)

    # ROC Curve (multiclass)
    st.subheader("ROC Curve")
    n_classes = len(np.unique(y))
    plot_multiclass_roc_curve(model, X_test, y_test, n_classes)

    # Precision-Recall Curve (multiclass)
    st.subheader("Precision-Recall Curve")
    plot_multiclass_precision_recall_curve(model, X_test, y_test, n_classes)

if __name__ == "__main__":
    main()
