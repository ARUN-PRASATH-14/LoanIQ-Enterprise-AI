import sys
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Reload trigger - rebuild index
load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.risk_agent import risk_agent
from agents.policy_agent import policy_agent
from agents.fraud_agent import fraud_agent
from agents.recommendation_agent import recommendation_agent
from database.mongo_client import (
    loan_applications_collection, prediction_history_collection,
    fraud_reports_collection, agent_logs_collection
)
from mcp_tools.tools import calculate_dti, calculate_pti, calculate_emi, evaluate_bank_rules

app = FastAPI(title="LoanIQ Enterprise AI Copilot API")

class LoanApplication(BaseModel):
    person_income: float
    loan_amnt: float
    person_emp_length: float
    loan_int_rate: float
    cb_person_cred_hist_length: float
    bank: str = "State Bank of India (SBI)"
    loan_type: str = "Home Loan"

@app.post("/predict")
def predict_loan(app_data: LoanApplication):
    data = app_data.model_dump()
    
    # Run hard rules via MCP tools
    dti_res = calculate_dti(data["person_income"], data["loan_amnt"])
    emi = calculate_emi(data["loan_amnt"], data["loan_int_rate"])
    pti_res = calculate_pti(data["person_income"], emi)
    
    data["dti_ratio"] = dti_res["dti_ratio"]
    data["monthly_payment"] = emi
    data["pti_ratio"] = pti_res["pti_ratio"]
    
    # Evaluate specific bank & scheme rules
    bank_eval = evaluate_bank_rules(
        data.get("bank", "SBI"),
        data.get("loan_type", "Home Loan"),
        data["person_income"],
        data["loan_amnt"],
        data["person_emp_length"],
        data["loan_int_rate"],
        data["cb_person_cred_hist_length"]
    )
    
    hard_rejected = not bank_eval["passed"]

    # ML Prediction
    risk_res = risk_agent.analyze(data)
    
    final_approved = risk_res["approved"] and not hard_rejected
    
    # Save to MongoDB
    record = {
        "timestamp": datetime.now(),
        "input_data": data,
        "risk_result": risk_res,
        "bank_evaluation": bank_eval
    }
    try:
        prediction_history_collection.insert_one(record)
    except Exception:
        pass
        
    return {
        "approved": final_approved,
        "risk_data": risk_res,
        "bank_evaluation": bank_eval,
        "hard_rejected": hard_rejected
    }

@app.post("/shap")
def get_shap(app_data: LoanApplication):
    data = app_data.model_dump()
    # Dummy features needed for scaled df inside agent
    data["dti_ratio"] = calculate_dti(data["person_income"], data["loan_amnt"])["dti_ratio"]
    emi = calculate_emi(data["loan_amnt"], data["loan_int_rate"])
    data["monthly_payment"] = emi
    data["pti_ratio"] = calculate_pti(data["person_income"], emi)["pti_ratio"]
    
    risk_res = risk_agent.analyze(data)
    return risk_res

@app.post("/rag")
def get_policy(app_data: LoanApplication):
    data = app_data.model_dump()
    # Adding mock DTI
    data["dti_ratio"] = calculate_dti(data["person_income"], data["loan_amnt"])["dti_ratio"]
    res = policy_agent.analyze(data)
    return res

@app.post("/fraud")
def get_fraud(app_data: LoanApplication):
    data = app_data.model_dump()
    res = fraud_agent.analyze(data)
    
    record = {
        "timestamp": datetime.now(),
        "input_data": data,
        "fraud_result": res
    }
    try:
        fraud_reports_collection.insert_one(record)
    except Exception:
        pass
        
    return res

@app.post("/recommendation")
def get_recommendation(app_data: LoanApplication):
    data = app_data.model_dump()
    data["dti_ratio"] = calculate_dti(data["person_income"], data["loan_amnt"])["dti_ratio"]
    emi = calculate_emi(data["loan_amnt"], data["loan_int_rate"])
    data["monthly_payment"] = emi
    data["pti_ratio"] = calculate_pti(data["person_income"], emi)["pti_ratio"]
    
    risk_res = risk_agent.analyze(data)
    policy_res = policy_agent.analyze(data)
    fraud_res = fraud_agent.analyze(data)
    
    rec_res = recommendation_agent.analyze(risk_res, policy_res, fraud_res)
    
    try:
        agent_logs_collection.insert_one({
            "timestamp": datetime.now(),
            "recommendation": rec_res
        })
    except Exception:
        pass
        
    return rec_res

from rag.vector_store import rag_system

class ChatQuery(BaseModel):
    query: str

@app.post("/chat")
def chat_query(req: ChatQuery):
    user_query = req.query.strip()
    clean_query = user_query.lower().strip("!.,? ")
    
    greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "hi there", "hello there", "help", "who are you"]
    if clean_query in greetings:
        return {
            "answer": "Hello! 👋 I am your **LoanIQ AI Policy & Banking Assistant**. I can help you with loan policies, interest rates, EMI calculations, terms & conditions, foreclosure charges, and eligibility rules for SBI, HDFC, ICICI, IOB, Canara Bank, and more! What would you like to know today?",
            "context": []
        }

    retrieved_chunks = rag_system.query(user_query, top_k=3)
    
    if not retrieved_chunks:
        return {"answer": "No relevant policy information found in the knowledge base.", "context": []}
        
    context_text = "\n\n".join(retrieved_chunks)
    
    import requests
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    API_URL = "https://api-inference.huggingface.co/models/HuggingFaceTB/SmolLM2-1.7B-Instruct"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    prompt = f"""You are an AI Banking Policy Assistant. Answer the question based ONLY on the following official bank policy documents.

Policy Documents:
{context_text}

Question: {user_query}
Answer:"""

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 200, "temperature": 0.3, "return_full_text": False}
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=8)
        if response.status_code == 200:
            answer = response.json()[0]['generated_text'].strip()
        else:
            answer = f"Based on retrieved Bank Guidelines:\n" + "\n\n".join([f"• {c}" for c in retrieved_chunks])
    except Exception:
        answer = f"Based on retrieved Bank Guidelines:\n" + "\n\n".join([f"• {c}" for c in retrieved_chunks])

    return {"answer": answer, "context": retrieved_chunks}