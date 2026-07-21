import streamlit as st
import requests
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Add root directory to sys.path for direct agent imports (Cloud deployment fallback)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.risk_agent import risk_agent
from agents.policy_agent import policy_agent
from agents.fraud_agent import fraud_agent
from agents.recommendation_agent import recommendation_agent
from mcp_tools.tools import calculate_dti, calculate_pti, calculate_emi, evaluate_bank_rules
from rag.vector_store import rag_system

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="LoanIQ Enterprise AI Copilot",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.stApp { background: #0a0e1a; color: #e8eaf6; }
#MainMenu, footer { visibility: hidden; }

.sidebar-content { background: #111827; }

/* Metrics and Cards */
.metric-card {
    background: rgba(17, 24, 39, 0.7);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(30, 45, 74, 0.8);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 3px; border-radius: 16px 16px 0 0;
}
.metric-card.blue::before  { background: linear-gradient(90deg,#3b82f6,#60a5fa); }
.metric-card.green::before { background: linear-gradient(90deg,#10b981,#34d399); }
.metric-card.red::before   { background: linear-gradient(90deg,#ef4444,#f87171); }
.metric-card.amber::before { background: linear-gradient(90deg,#f59e0b,#fbbf24); }

.metric-label { font-size:.7rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:#64748b; margin-bottom:.3rem; }
.metric-value { font-family:'JetBrains Mono',monospace; font-size:1.8rem; font-weight:700; line-height:1; margin-bottom:.2rem; }
.metric-value.blue  { color:#60a5fa; }
.metric-value.green { color:#34d399; }
.metric-value.red   { color:#f87171; }
.metric-value.amber { color:#fbbf24; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.5rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(59,130,246,0.5) !important;
}

/* Titles */
.app-title {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg,#60a5fa,#a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0;
}

.section-title {
    font-size:.8rem; font-weight:700; letter-spacing:.12em;
    text-transform:uppercase; color:#3b82f6;
    margin-bottom:.85rem; padding-bottom:.4rem; border-bottom:1px solid #1e2d4a;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="app-title">🏦 LoanIQ Enterprise AI Copilot</p>', unsafe_allow_html=True)
st.markdown('<p style="color:#64748b; margin-bottom: 1rem;">AI-powered Loan Approval, Risk Analysis, Explainable AI, and Policy Intelligence</p>', unsafe_allow_html=True)

# Helper function to execute AI Engine (with Cloud fallback)
def run_ai_analysis(payload):
    try:
        # Try FastAPI Microservice endpoint first
        pred_resp = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=1.5).json()
        shap_resp = requests.post(f"{API_BASE_URL}/shap", json=payload, timeout=1.5).json()
        rag_resp = requests.post(f"{API_BASE_URL}/rag", json=payload, timeout=1.5).json()
        fraud_resp = requests.post(f"{API_BASE_URL}/fraud", json=payload, timeout=1.5).json()
        rec_resp = requests.post(f"{API_BASE_URL}/recommendation", json=payload, timeout=1.5).json()
        return pred_resp, shap_resp, rag_resp, fraud_resp, rec_resp
    except Exception:
        # Cloud/Standalone Direct Agent Execution Fallback
        income = payload["person_income"]
        loan = payload["loan_amnt"]
        emp = payload["person_emp_length"]
        rate = payload["loan_int_rate"]
        credit = payload["cb_person_cred_hist_length"]
        bank = payload["bank"]
        loan_type = payload["loan_type"]
        
        dti_res = calculate_dti(income, loan)
        emi = calculate_emi(loan, rate)
        pti_res = calculate_pti(income, emi)
        
        data = dict(payload)
        data["dti_ratio"] = dti_res["dti_ratio"]
        data["monthly_payment"] = emi
        data["pti_ratio"] = pti_res["pti_ratio"]
        
        bank_eval = evaluate_bank_rules(bank, loan_type, income, loan, emp, rate, credit)
        hard_rejected = not bank_eval["passed"]
        
        risk_res = risk_agent.analyze(data)
        policy_res = policy_agent.analyze(data)
        fraud_res = fraud_agent.analyze(data)
        rec_res = recommendation_agent.analyze(risk_res, policy_res, fraud_res)
        
        final_approved = risk_res["approved"] and not hard_rejected
        
        pred_resp = {
            "approved": final_approved,
            "risk_data": risk_res,
            "bank_evaluation": bank_eval,
            "hard_rejected": hard_rejected
        }
        return pred_resp, risk_res, policy_res, fraud_res, rec_res

def run_chat_query(query_text):
    try:
        return requests.post(f"{API_BASE_URL}/chat", json={"query": query_text}, timeout=1.5).json()
    except Exception:
        clean_query = query_text.strip().lower().strip("!.,? ")
        greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "hi there", "hello there", "help", "who are you"]
        if clean_query in greetings:
            return {
                "answer": "Hello! 👋 I am your **LoanIQ AI Policy & Banking Assistant**. I can help you with loan policies, interest rates, EMI calculations, terms & conditions, foreclosure charges, and eligibility rules for SBI, HDFC, ICICI, IOB, Canara Bank, and more! What would you like to know today?",
                "context": []
            }
            
        retrieved_chunks = rag_system.query(query_text, top_k=3)
        if not retrieved_chunks:
            return {"answer": "No relevant policy information found in the knowledge base.", "context": []}
            
        context_text = "\n\n".join(retrieved_chunks)
        HF_TOKEN = os.getenv("HF_TOKEN", "")
        API_URL = "https://api-inference.huggingface.co/models/HuggingFaceTB/SmolLM2-1.7B-Instruct"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        
        prompt = f"You are an AI Banking Policy Assistant. Answer the question based ONLY on the following official bank policy documents.\n\nPolicy Documents:\n{context_text}\n\nQuestion: {query_text}\nAnswer:"
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200, "temperature": 0.3, "return_full_text": False}}
        
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=8)
            if response.status_code == 200:
                answer = response.json()[0]['generated_text'].strip()
            else:
                answer = f"Based on retrieved Bank Guidelines:\n" + "\n\n".join([f"• {c}" for c in retrieved_chunks])
        except Exception:
            answer = f"Based on retrieved Bank Guidelines:\n" + "\n\n".join([f"• {c}" for c in retrieved_chunks])
            
        return {"answer": answer, "context": retrieved_chunks}

# Main Navigation Tabs
tab1, tab2 = st.tabs(["📊 Decision Engine & Risk Copilot", "💬 Policy Intelligence RAG Chatbot"])

# Sidebar Inputs
with st.sidebar:
    st.markdown('<div class="section-title">🏛️ Bank & Scheme Selection</div>', unsafe_allow_html=True)
    selected_bank = st.selectbox(
        "Select Target Bank",
        [
            "State Bank of India (SBI)",
            "HDFC Bank",
            "ICICI Bank",
            "Indian Overseas Bank (IOB)",
            "Canara Bank"
        ]
    )
    selected_scheme = st.selectbox(
        "Loan Category / Scheme",
        [
            "Home Loan",
            "Personal Loan",
            "Education Loan",
            "Agriculture / Farmer Loan (KCC)"
        ]
    )
    
    st.markdown('<div class="section-title" style="margin-top:1rem;">📝 Application Data</div>', unsafe_allow_html=True)
    income = st.number_input("Annual Income ($)", min_value=10000, max_value=5000000, value=75000, step=1000)
    loan = st.number_input("Loan Amount ($)", min_value=1000, max_value=5000000, value=25000, step=1000)
    emp_len = st.number_input("Employment (Years)", min_value=0.0, max_value=40.0, value=4.0, step=0.5)
    rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=40.0, value=11.5, step=0.5)
    credit_len = st.number_input("Credit History (Years)", min_value=0, max_value=40, value=5, step=1)
    
    analyze_btn = st.button("🚀 Run Enterprise AI Analysis")

with tab1:
    if analyze_btn:
        payload = {
            "person_income": income,
            "loan_amnt": loan,
            "person_emp_length": emp_len,
            "loan_int_rate": rate,
            "cb_person_cred_hist_length": credit_len,
            "bank": selected_bank,
            "loan_type": selected_scheme
        }
        
        with st.spinner("Analyzing Risk Profile..."):
            pred_resp, shap_resp, rag_resp, fraud_resp, rec_resp = run_ai_analysis(payload)
            
            # 1. Top Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            hard_rej = pred_resp.get("hard_rejected", False)
            approved = pred_resp.get("approved", False)
            
            risk_data = pred_resp.get("risk_data", {})
            prob = risk_data.get("probability", 0)
            
            status_color = "green" if approved else "red"
            status_text = "APPROVED" if approved else "REJECTED"
            
            col1.markdown(f'<div class="metric-card {status_color}"><div class="metric-label">Decision</div><div class="metric-value {status_color}">{status_text}</div></div>', unsafe_allow_html=True)
            
            prob_color = "green" if prob >= 0.5 else "red"
            col2.markdown(f'<div class="metric-card {prob_color}"><div class="metric-label">AI Approval Prob</div><div class="metric-value {prob_color}">{prob*100:.1f}%</div></div>', unsafe_allow_html=True)
            
            fraud_risk = fraud_resp.get("risk_level", "UNKNOWN")
            fraud_col = "green" if fraud_risk == "LOW" else "amber" if fraud_risk == "MEDIUM" else "red"
            col3.markdown(f'<div class="metric-card {fraud_col}"><div class="metric-label">Fraud Risk</div><div class="metric-value {fraud_col}">{fraud_risk}</div></div>', unsafe_allow_html=True)
            
            dti = loan / (income + 1)
            dti_col = "green" if dti <= 0.43 else "red"
            col4.markdown(f'<div class="metric-card {dti_col}"><div class="metric-label">DTI Ratio</div><div class="metric-value {dti_col}">{dti*100:.1f}%</div></div>', unsafe_allow_html=True)
            
            bank_eval = pred_resp.get("bank_evaluation", {})
            violations = bank_eval.get("violations", [])
            if violations:
                st.warning(f"⚠️ **{selected_bank} Policy Violations:** " + " | ".join(violations))
            
            # 2. Main Content Area
            m1, m2 = st.columns([1, 1])
            
            with m1:
                st.markdown('<div class="section-title">🧠 Explainable AI (SHAP)</div>', unsafe_allow_html=True)
                top_factors = shap_resp.get("top_factors", [])
                for f in top_factors:
                    val = f["contribution"]
                    col = "#10b981" if val > 0 else "#ef4444"
                    st.markdown(f"""
                    <div style="background:rgba(30,45,74,0.3); padding:10px; border-radius:8px; margin-bottom:8px; border-left: 4px solid {col};">
                        <strong>{f['feature']}</strong>: {val:+.3f} contribution
                    </div>
                    """, unsafe_allow_html=True)
                    
            with m2:
                st.markdown('<div class="section-title">🤖 SLM Recommendation</div>', unsafe_allow_html=True)
                rec_text = rec_resp.get("recommendation", "No recommendation generated.")
                st.info(rec_text)
                
                st.markdown(f'<div class="section-title" style="margin-top: 1rem;">📚 Policy Evidence — {selected_bank} ({selected_scheme})</div>', unsafe_allow_html=True)
                evidence = rag_resp.get("evidence", [])
                if not evidence:
                    st.write("No specific policy violations found.")
                for ev in evidence:
                    st.markdown(f"""<div style="font-size: 0.85rem; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 6px; margin-bottom: 8px;">{ev}</div>""", unsafe_allow_html=True)
    else:
        st.info("👈 Adjust the sidebar inputs and click **'Run Enterprise AI Analysis'** to evaluate a loan application.")

with tab2:
    st.markdown('<div class="section-title">💬 Policy Intelligence & Guidelines Chatbot</div>', unsafe_allow_html=True)
    st.markdown("Ask any questions regarding loan interest rates, RBI rules, SBI, HDFC, ICICI, IOB, or Canara Bank policies.")
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hello! 👋 I am your AI Policy Intelligence Assistant. Ask me anything about Indian bank policies, interest rates, DTI rules, or collateral requirements for SBI, HDFC, ICICI, IOB, or Canara Bank!"}
        ]
        
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    user_query = st.chat_input("e.g. What is the SBI Kisan Credit Card interest rate?")
    if user_query:
        st.session_state.chat_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        with st.chat_message("assistant"):
            with st.spinner("Searching FAISS Policy Vector DB..."):
                try:
                    chat_resp = run_chat_query(user_query)
                    ans = chat_resp.get("answer", "No answer found.")
                    context_chunks = chat_resp.get("context", [])
                    
                    st.markdown(ans)
                    if context_chunks:
                        with st.expander("🔍 View Retrieved FAISS Policy Evidence Chunks"):
                            for chunk in context_chunks:
                                st.markdown(f"- {chunk}")
                                
                    st.session_state.chat_messages.append({"role": "assistant", "content": ans})
                except Exception as e:
                    err_msg = f"Error querying policy assistant: {str(e)}"
                    st.error(err_msg)
                    st.session_state.chat_messages.append({"role": "assistant", "content": err_msg})
