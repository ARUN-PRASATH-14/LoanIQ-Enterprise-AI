import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mcp_tools.tools import check_fraud

class FraudAgent:
    def __init__(self):
        pass
        
    def analyze(self, applicant_data: dict):
        income = applicant_data.get('person_income', 0)
        loan_amount = applicant_data.get('loan_amnt', 0)
        emp_length = applicant_data.get('person_emp_length', 0)
        
        result = check_fraud(income, loan_amount, emp_length)
        return result

fraud_agent = FraudAgent()
