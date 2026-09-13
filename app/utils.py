from pathlib import Path
import joblib
import numpy as np
import pandas as pd

SERVICES = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]

def load_artifacts():
    """Load serialized model, metadata, and SHAP explainer with caching safety."""
    root_dir = Path(__file__).resolve().parent.parent
    model_path = root_dir / "models" / "churn_model.pkl"
    metadata_path = root_dir / "models" / "model_metadata.pkl"
    explainer_path = root_dir / "models" / "shap_explainer.pkl"
    test_data_path = root_dir / "data" / "X_test.csv"
    test_target_path = root_dir / "data" / "y_test.csv"

    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    explainer = joblib.load(explainer_path)
    X_test = pd.read_csv(test_data_path)
    y_test = pd.read_csv(test_target_path).squeeze("columns")

    return model, metadata, explainer, X_test, y_test

def encode_input_data(user_inputs: dict, feature_names: list) -> pd.DataFrame:
    """
    Transforms raw UI form inputs into the exact 35-column one-hot encoded vector.
    Initializes a zero-filled template to avoid missing category mismatches.
    """
    # 1. Initialize empty DataFrame with expected 35 features
    df_encoded = pd.DataFrame(0.0, index=[0], columns=feature_names)

    # 2. Derive behavioral features
    tenure = float(user_inputs["tenure"])
    monthly_charges = float(user_inputs["MonthlyCharges"])
    total_charges = float(user_inputs["TotalCharges"])

    # AvgMonthlySpend
    avg_monthly_spend = total_charges / tenure if tenure > 0 else monthly_charges

    # ServiceCount
    service_count = sum(1 for s in SERVICES if user_inputs.get(s) == "Yes")

    # TenureGroup allocation
    if tenure <= 12:
        tenure_group = "0-12m"
    elif tenure <= 24:
        tenure_group = "13-24m"
    elif tenure <= 48:
        tenure_group = "25-48m"
    else:
        tenure_group = "49+m"

    # 3. Assign direct numeric values
    df_encoded.at[0, "SeniorCitizen"] = float(user_inputs["SeniorCitizen"])
    df_encoded.at[0, "tenure"] = tenure
    df_encoded.at[0, "MonthlyCharges"] = monthly_charges
    df_encoded.at[0, "TotalCharges"] = total_charges
    df_encoded.at[0, "ServiceCount"] = float(service_count)
    df_encoded.at[0, "AvgMonthlySpend"] = float(avg_monthly_spend)

    # 4. Populate one-hot dummy flags matching get_dummies(drop_first=True)
    dummy_mappings = {
        f"gender_{user_inputs['gender']}": 1.0,
        f"Partner_{user_inputs['Partner']}": 1.0,
        f"Dependents_{user_inputs['Dependents']}": 1.0,
        f"PhoneService_{user_inputs['PhoneService']}": 1.0,
        f"MultipleLines_{user_inputs['MultipleLines']}": 1.0,
        f"InternetService_{user_inputs['InternetService']}": 1.0,
        f"OnlineSecurity_{user_inputs['OnlineSecurity']}": 1.0,
        f"OnlineBackup_{user_inputs['OnlineBackup']}": 1.0,
        f"DeviceProtection_{user_inputs['DeviceProtection']}": 1.0,
        f"TechSupport_{user_inputs['TechSupport']}": 1.0,
        f"StreamingTV_{user_inputs['StreamingTV']}": 1.0,
        f"StreamingMovies_{user_inputs['StreamingMovies']}": 1.0,
        f"Contract_{user_inputs['Contract']}": 1.0,
        f"PaperlessBilling_{user_inputs['PaperlessBilling']}": 1.0,
        f"PaymentMethod_{user_inputs['PaymentMethod']}": 1.0,
        f"TenureGroup_{tenure_group}": 1.0
    }

    for col, val in dummy_mappings.items():
        if col in df_encoded.columns:
            df_encoded.at[0, col] = val

    return df_encoded

def get_retention_recommendations(user_inputs: dict, proba: float, threshold: float) -> list:
    """Generates prescriptive, rule-based retention actions based on customer risk."""
    offers = []
    if proba < threshold:
        return ["Customer risk is low. Maintain regular engagement and standard loyalty rewards."]

    if user_inputs.get("Contract") == "Month-to-month":
        offers.append("Offer a 1-year contract extension with a 15% promotional discount on monthly charges.")
    if user_inputs.get("PaymentMethod") == "Electronic check":
        offers.append("Incentivize enrollment in automatic bank/credit card payments with a $10 one-time billing credit.")
    if user_inputs.get("TechSupport") == "No" and user_inputs.get("InternetService") != "No":
        offers.append("Provide a complimentary 6-month trial of TechSupport and OnlineSecurity services.")
    if user_inputs.get("tenure") <= 12:
        offers.append("Schedule an onboarding success call to resolve early account or service friction.")
    if not offers:
        offers.append("Assign customer to premium retention desk for personalized outbound review.")

    return offers