import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
print("🔑 API Key:", api_key)

# Configure API
genai.configure(api_key=api_key)

# Use one of the working models
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

# Generate content
response = model.generate_content("தமிழ் நாட்டில் தக்காளி வளர்க்க சிறந்த வழிகள் என்ன?")
print(response.text)
