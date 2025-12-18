import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

st.set_page_config(page_title="Fraud Detection System", layout="wide", page_icon="🛡️")

@st.cache_resource
def load_models():
    with open("data/processed/xgboost_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("data/processed/xgb_threshold.pkl", "rb") as f:
        threshold = pickle.load(f)
    return xgb_model, threshold

@st.cache_data
def load_data():
    X_test = pd.read_pickle("data/processed/X_test.pkl")
    y_test = pd.read_pickle("data/processed/y_test.pkl")
    return X_test, y_test

xgb_model, threshold = load_models()
X_test, y_test = load_data()

y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
total_frauds = y_test.sum()
detected_frauds = tp
false_positives = fp

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Dashboard", "Model Performance", "Transaction Analysis"])

if page == "Dashboard":
    st.title("Real-Time Fraud Detection System")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Total Transactions", value=f"{len(y_test):,}")
    
    with col2:
        detection_rate = (detected_frauds / total_frauds * 100) if total_frauds > 0 else 0
        st.metric(label="Frauds Detected", value=f"{detected_frauds}/{total_frauds}", delta=f"{detection_rate:.1f}%")
    
    with col3:
        fp_rate = (false_positives / len(y_test) * 100)
        st.metric(label="False Positive Rate", value=f"{fp_rate:.3f}%", delta=f"{false_positives} cases")
    
    with col4:
        savings = detected_frauds * 122 - false_positives * 5
        st.metric(label="Estimated Savings", value=f"${savings:,.0f}", delta="ROI positive")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=["Legitimate", "Fraud"], y=["Legitimate", "Fraud"],
                       color_continuous_scale="Blues", text_auto=True)
        fig.update_layout(title="Confusion Matrix", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=y_pred_proba[y_test == 0], name="Legitimate", opacity=0.7, marker_color="green"))
        fig.add_trace(go.Histogram(x=y_pred_proba[y_test == 1], name="Fraud", opacity=0.7, marker_color="red"))
        fig.add_vline(x=threshold, line_dash="dash", line_color="black", annotation_text=f"Threshold: {threshold:.3f}")
        fig.update_layout(title="Fraud Score Distribution", xaxis_title="Fraud Probability",
                         yaxis_title="Count", barmode="overlay", height=400)
        st.plotly_chart(fig, use_container_width=True)

elif page == "Model Performance":
    st.title("Model Performance Analysis")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve", line=dict(color="blue", width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="gray", dash="dash")))
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", line=dict(color="green", width=2)))
        fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        "feature": X_test.columns,
        "importance": xgb_model.feature_importances_
    }).sort_values("importance", ascending=False).head(15)
    
    fig = px.bar(feature_importance, x="importance", y="feature", orientation="h", title="Top 15 Most Important Features")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

elif page == "Transaction Analysis":
    st.title("Transaction-Level Analysis")
    st.markdown("---")
    
    filter_type = st.selectbox("Show:", ["All Frauds", "Detected Frauds", "Missed Frauds", "False Positives"])
    
    if filter_type == "All Frauds":
        indices = y_test[y_test == 1].index
    elif filter_type == "Detected Frauds":
        indices = y_test[(y_test == 1) & (y_pred == 1)].index
    elif filter_type == "Missed Frauds":
        indices = y_test[(y_test == 1) & (y_pred == 0)].index
    else:
        indices = y_test[(y_test == 0) & (y_pred == 1)].index
    
    df_display = X_test.loc[indices].copy()
    df_display["True_Label"] = y_test.loc[indices]
    df_display["Predicted_Label"] = y_pred[y_test.index.isin(indices)]
    df_display["Fraud_Score"] = y_pred_proba[y_test.index.isin(indices)]
    
    st.write(f"Total transactions: {len(df_display)}")
    st.dataframe(df_display[["Amount", "Time_hours", "Fraud_Score", "True_Label", "Predicted_Label"]].head(50))

st.sidebar.markdown("---")
st.sidebar.info("Fraud Detection System v1.0")
