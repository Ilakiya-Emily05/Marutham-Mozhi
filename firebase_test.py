from firebase_helper import save_prediction
import datetime

# Test values
user_id = "test_user"
disease = "Tomato___Late_blight"
timestamp = str(datetime.datetime.now())

save_prediction(user_id, disease, timestamp)

print("✅ Test data sent to Firebase!")
