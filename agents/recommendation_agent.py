import os
import requests

HF_TOKEN = os.getenv("HF_TOKEN", "")
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceTB/SmolLM2-1.7B-Instruct"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

class RecommendationAgent:
    def __init__(self):
        pass
        
    def analyze(self, risk_data, policy_data, fraud_data):
        prompt = f"""You are a Senior Bank Loan Officer Assistant.
Write a 3-4 sentence professional recommendation for this loan application based on the following data.
Do NOT predict or guess, just summarize these facts.

Risk Data: 
Probability of Approval: {risk_data.get('probability', 0)*100:.1f}%
Top Factors: {[f['feature'] for f in risk_data.get('top_factors', [])[:3]]}

Fraud Data:
Fraud Risk: {fraud_data.get('risk_level', 'Unknown')}
Reasons: {fraud_data.get('reasons', [])}

Policy Evidence:
{policy_data.get('evidence', [])}

Recommendation:"""
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.2,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()[0]['generated_text']
                return {"recommendation": result.strip()}
            else:
                return {"recommendation": "Unable to generate recommendation due to API error."}
        except requests.exceptions.ConnectionError:
            # Fallback if there is no internet connection or DNS issue
            fallback = f"Recommendation: The applicant has a {fraud_data.get('risk_level', 'Unknown')} fraud risk and an AI approval probability of {risk_data.get('probability', 0)*100:.1f}%. Please review the DTI and PTI ratios manually against bank guidelines."
            return {"recommendation": fallback}
        except Exception as e:
            return {"recommendation": "Error: Could not reach the recommendation API at this time."}

recommendation_agent = RecommendationAgent()
