import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from utils import encode_input_data, get_retention_recommendations, load_artifacts

# Page configuration
st.set_page_config(
    page_title="Customer Churn Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

# Cached asset loader
@st.cache_resource
def get_cached_assets():
    return load_artifacts()

try:
    model, metadata, explainer, X_test, y_test = get_cached_assets()
    feature_names = metadata["feature_names"]
    threshold = metadata.get("optimal_threshold", 0.35)
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Workspace:",
    ["Risk Scoring & Local SHAP", "Portfolio Churn Analytics"]
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Calibrated Cutoff:** `{threshold:.2f}`")
st.sidebar.markdown(f"**Total Features:** `{len(feature_names)}`")

# -------------------------------------------------------------
# PAGE 1: Risk Scoring & Local Explainability
# -------------------------------------------------------------
if page == "Risk Scoring & Local SHAP":
    st.title("🎯 Customer Risk Scoring & Explainability")
    st.markdown("Score real-time customer profiles, inspect attrition drivers, and trigger retention interventions.")

    with st.form("customer_input_form"):
        st.subheader("Account & Demographic Attributes")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        with c2:
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
        with c3:
            tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=3)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        with c4:
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        st.subheader("Subscribed Services & Financials")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        with s2:
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        with s3:
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        with s4:
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])

        st.markdown("---")
        f1, f2, f3 = st.columns(3)
        with f1:
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        with f2:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=85.0, step=1.0)
        with f3:
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=255.0, step=10.0)

        submitted = st.form_submit_button("Predict & Explain Risk", use_container_width=True)

    if submitted:
        raw_inputs = {
            "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet_service,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        # Encode input
        encoded_payload = encode_input_data(raw_inputs, feature_names)

        # Inference
        churn_prob = float(model.predict_proba(encoded_payload)[0, 1])
        is_churn = churn_prob >= threshold

        # Risk Tier
        if churn_prob < 0.30:
            tier, color = "LOW RISK", "normal"
        elif churn_prob < 0.60:
            tier, color = "MEDIUM RISK", "off"
        else:
            tier, color = "HIGH RISK", "inverse"

        st.markdown("### Risk Assessment")
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Churn Probability", f"{churn_prob:.1%}")
        m2.metric("Decision Threshold", f"{threshold:.2f}")
        m3.metric("System Recommendation", "FLAGGED FOR RETENTION" if is_churn else "RETAINED / STABLE")

        st.markdown("---")
        exp_col, rec_col = st.columns([3, 2])

        with exp_col:
            st.subheader("🔍 Local SHAP Feature Attribution")
            shap_explanation = explainer(encoded_payload)
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(shap_explanation[0], max_display=8, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with rec_col:
            st.subheader("📋 Recommended Interventions")
            recommendations = get_retention_recommendations(raw_inputs, churn_prob, threshold)
            for rec in recommendations:
                st.info(f"• {rec}")

# -------------------------------------------------------------
# PAGE 2: Portfolio Churn Analytics
# -------------------------------------------------------------
elif page == "Portfolio Churn Analytics":
    st.title("📈 Portfolio Churn Analytics")
    st.markdown("Macro-level retention KPIs and segment breakdowns computed across the 1,409 held-out test accounts.")

    # Calculate portfolio statistics
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= threshold).astype(int)

    total_customers = len(X_test)
    at_risk_count = int(test_preds.sum())
    captured_churners = int(((test_preds == 1) & (y_test.values == 1)).sum())
    total_actual_churn = int(y_test.sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Test Pool Accounts", f"{total_customers:,}")
    k2.metric("Flagged High-Risk", f"{at_risk_count:,}", f"{(at_risk_count/total_customers):.1%}")
    k3.metric("Captured Real Churners", f"{captured_churners} / {total_actual_churn}")
    k4.metric("Minority Recall", f"{(captured_churners/total_actual_churn):.1%}")

    st.markdown("---")
    g1, g2 = st.columns(2)

    with g1:
        st.subheader("Risk Score Distribution")
        fig_hist, ax_hist = plt.subplots(figsize=(8, 4.5))
        ax_hist.hist(test_probs, bins=25, color="#1f77b4", edgecolor="black", alpha=0.7)
        ax_hist.axvline(threshold, color="red", linestyle="--", label=f"Cutoff ({threshold:.2f})")
        ax_hist.set_xlabel("Predicted Churn Probability")
        ax_hist.set_ylabel("Customer Count")
        ax_hist.legend()
        st.pyplot(fig_hist)
        plt.close(fig_hist)

    with g2:
        st.subheader("Global Feature Importance (Top 10)")
        feature_importance = pd.Series(
            model.feature_importances_, index=feature_names
        ).sort_values(ascending=True).tail(10)

        fig_imp, ax_imp = plt.subplots(figsize=(8, 4.5))
        feature_importance.plot(kind="barh", ax=ax_imp, color="#2ca02c")
        ax_imp.set_xlabel("XGBoost Split Importance (Gain)")
        st.pyplot(fig_imp)
        plt.close(fig_imp)