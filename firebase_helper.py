import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase app only once
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)

# Firestore client
db = firestore.client()

# Save prediction to Firestore
def save_prediction(user_id, disease, timestamp):
    try:
        doc_ref = db.collection("predictions").document()
        doc_ref.set({
            "user_id": user_id,
            "disease": disease,
            "timestamp": timestamp
        })
        print("✅ Prediction saved to Firebase.")
    except Exception as e:
        print(f"❌ Failed to save to Firebase: {e}")
