import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rag.vector_store import rag_system

class PolicyAgent:
    def __init__(self):
        self.rag = rag_system
        
    def analyze(self, applicant_data: dict):
        # Construct queries based on applicant data, selected bank and loan scheme
        dti = applicant_data.get('dti_ratio', 0)
        loan = applicant_data.get('loan_amnt', 0)
        bank = applicant_data.get('bank', 'SBI')
        loan_type = applicant_data.get('loan_type', 'Home Loan')
        
        # Clean bank name for query matching (e.g. 'State Bank of India (SBI)' -> 'SBI')
        bank_keyword = "SBI" if "SBI" in bank else "HDFC" if "HDFC" in bank else "ICICI" if "ICICI" in bank else "IOB" if "IOB" in bank else "Canara" if "Canara" in bank else bank
        
        queries = [
            f"{bank_keyword} {loan_type} policy terms interest rate maximum amount limit",
            f"{bank_keyword} {loan_type} conditions collateral credit history"
        ]
        
        if dti > 0.40:
            queries.append(f"{bank_keyword} Debt to income ratio DTI limit guidelines")
            
        evidence = []
        for q in queries:
            results = self.rag.query(q, top_k=1)
            for r in results:
                if r not in evidence:
                    evidence.append(r)
                    
        return {
            "evidence": evidence,
            "target_bank": bank,
            "loan_scheme": loan_type
        }

policy_agent = PolicyAgent()
