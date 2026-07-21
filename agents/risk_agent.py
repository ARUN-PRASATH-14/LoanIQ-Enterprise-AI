import pickle
import numpy as np
import pandas as pd
import shap
import os

model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'loan_model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')
features_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_names.pkl')
threshold_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'threshold.pkl')

class RiskAgent:
    def __init__(self):
        self.model = pickle.load(open(model_path, "rb"))
        self.scaler = pickle.load(open(scaler_path, "rb"))
        self.feature_names = pickle.load(open(features_path, "rb"))
        self.threshold = pickle.load(open(threshold_path, "rb"))
        self.explainer = shap.TreeExplainer(self.model)

    def analyze(self, applicant_data: dict):
        # Compute engineered features
        income = applicant_data.get('person_income', 0)
        loan = applicant_data.get('loan_amnt', 0)
        emp = applicant_data.get('person_emp_length', 0)
        rate = applicant_data.get('loan_int_rate', 0)
        credit = applicant_data.get('cb_person_cred_hist_length', 0)
        
        applicant_data["dti_ratio"] = loan / (income + 1)
        applicant_data["monthly_payment"] = (loan * rate / 100) / 12
        applicant_data["pti_ratio"] = applicant_data["monthly_payment"] / (income / 12 + 1)
        applicant_data["income_loan_ratio"] = income / (loan + 1)
        applicant_data["interest_income_burden"] = rate * loan / (income + 1)
        applicant_data["emp_credit_ratio"] = emp / (credit + 1)
        applicant_data["annual_interest_cost"] = loan * rate / 100
        applicant_data["loan_per_emp_year"] = loan / (emp + 1)

        df = pd.DataFrame([applicant_data])[self.feature_names]
        df_scaled = self.scaler.transform(df)
        
        prob = float(self.model.predict_proba(df_scaled)[0][1])
        approved = prob >= self.threshold
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(df_scaled)
        
        # If binary classification, shap_values might be a list or single array
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]
            
        base_value = self.explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[1] if len(base_value)>1 else base_value[0])
        else:
            base_value = float(base_value)

        # Get top contributors
        feature_contributions = []
        for i, name in enumerate(self.feature_names):
            feature_contributions.append({
                "feature": name,
                "value": float(df.iloc[0, i]),
                "contribution": float(sv[i])
            })
            
        # Sort by absolute contribution
        feature_contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        return {
            "probability": prob,
            "threshold": self.threshold,
            "approved": approved,
            "shap_base_value": base_value,
            "shap_values": sv.tolist(),
            "top_factors": feature_contributions[:5]
        }

risk_agent = RiskAgent()
