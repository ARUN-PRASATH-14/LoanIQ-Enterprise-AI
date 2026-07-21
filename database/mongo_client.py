import os
from pymongo import MongoClient

# Provided connection string
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://arunpvt2005_db_user:I0xedwjQ9Du2fWbC@cluster0.1tgrswq.mongodb.net/?appName=Cluster0"
)

# Connect to MongoDB cluster
client = MongoClient(MONGO_URI)

# Database Name
db = client["loan_iq_enterprise"]

# Collections
customers_collection = db["customers"]
loan_applications_collection = db["loan_applications"]
prediction_history_collection = db["prediction_history"]
policy_documents_collection = db["policy_documents"]
fraud_reports_collection = db["fraud_reports"]
agent_logs_collection = db["agent_logs"]

def check_connection():
    try:
        # The ping command is cheap and does not require auth.
        client.admin.command('ping')
        return True
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        return False
