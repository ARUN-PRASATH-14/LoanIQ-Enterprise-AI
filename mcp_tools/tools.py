import json

def calculate_dti(income: float, loan_amount: float) -> dict:
    """Debt-To-Income Ratio Calculator."""
    dti = loan_amount / (income + 1)
    return {"dti_ratio": round(dti, 4), "status": "FAIL" if dti > 0.43 else "PASS"}

def calculate_pti(income: float, monthly_payment: float) -> dict:
    """Payment-To-Income Ratio Calculator."""
    pti = monthly_payment / ((income / 12) + 1)
    return {"pti_ratio": round(pti, 4), "status": "FAIL" if pti > 0.35 else "PASS"}

def calculate_emi(loan_amount: float, interest_rate: float) -> float:
    """EMI Calculator (rough estimate for annual interest)."""
    monthly_payment = (loan_amount * interest_rate / 100) / 12
    return round(monthly_payment, 2)

def check_fraud(income: float, loan_amount: float, emp_length: float) -> dict:
    """Mock Fraud Lookup Tool."""
    score = 0
    reasons = []
    
    if loan_amount > income * 5:
        score += 50
        reasons.append("Loan amount exceeds 5x annual income")
        
    if emp_length < 0.5 and loan_amount > 20000:
        score += 40
        reasons.append("High loan amount with very low employment length")
        
    if score >= 50:
        risk = "HIGH"
    elif score >= 20:
        risk = "MEDIUM"
    else:
        risk = "LOW"
        
    return {"fraud_score": score, "risk_level": risk, "reasons": reasons}

def evaluate_bank_rules(bank: str, loan_type: str, income: float, loan_amount: float, emp_length: float, interest_rate: float, credit_history: float) -> dict:
    """Evaluate specific bank & loan scheme rules."""
    violations = []
    
    dti = loan_amount / (income + 1)
    monthly_payment = (loan_amount * interest_rate / 100) / 12
    pti = monthly_payment / ((income / 12) + 1)
    
    bank_clean = "SBI" if "SBI" in bank else "HDFC" if "HDFC" in bank else "ICICI" if "ICICI" in bank else "IOB" if "IOB" in bank else "Canara" if "Canara" in bank else bank

    # Bank-Specific Checks
    if bank_clean == "SBI":
        if dti > 0.50:
            violations.append("SBI DTI exceeds maximum allowed threshold (50%)")
        if "Personal" in loan_type and loan_amount > 200000:
            violations.append("SBI Personal Loan exceeds maximum limit of $200,000")
        if "Personal" in loan_type and interest_rate < 11.15:
            violations.append("SBI Personal Loan interest rate below minimum standard (11.15%)")
            
    elif bank_clean == "HDFC":
        if income < 30000:
            violations.append("HDFC requires minimum annual income of $30,000")
        if dti > 0.45:
            violations.append("HDFC DTI exceeds 45% policy threshold")
        if "Personal" in loan_type and loan_amount > 400000:
            violations.append("HDFC Personal Loan exceeds $400,000 cap")
            
    elif bank_clean == "ICICI":
        if emp_length < 2.0:
            violations.append("ICICI requires minimum 2.0 years of employment history")
        if credit_history < 2.0:
            violations.append("ICICI requires minimum 2.0 years of credit history")
        if dti > 0.43:
            violations.append("ICICI DTI exceeds 43% cap")
            
    elif bank_clean == "IOB":
        if "Personal" in loan_type and loan_amount > (income * 1.25):
            violations.append("IOB Personal Loan exceeds 15x gross monthly salary (1.25x annual income)")
        if emp_length < 3.0 and "Personal" in loan_type:
            violations.append("IOB Personal Loan requires minimum 3 years of service")
            
    elif bank_clean == "Canara":
        if income < 15000:
            violations.append("Canara Bank requires minimum annual income of $15,000")
        if "Personal" in loan_type and loan_amount > 100000:
            violations.append("Canara Bank Personal Loan exceeds maximum $100,000 cap")

    # Global baseline checks
    if dti > 0.60:
        violations.append(f"{bank_clean} DTI ratio ({dti:.1%}) critically high (max 60%)")
    if pti > 0.40:
        violations.append(f"{bank_clean} PTI ratio ({pti:.1%}) exceeds affordability limit (40%)")

    passed = len(violations) == 0
    return {
        "passed": passed,
        "violations": violations,
        "bank": bank_clean,
        "loan_scheme": loan_type
    }
