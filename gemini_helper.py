import os
from dotenv import load_dotenv
import google.generativeai as genai
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("GEMINI_API_KEY")
print("🔑 GEMINI_API_KEY loaded:", API_KEY)  

genai.configure(api_key=API_KEY)

MODEL_NAME = "models/gemini-1.5-flash"  

def get_gemini_response(disease_name):
    prompt = f"""Explain the plant disease '{disease_name}' in simple terms 
    and give natural treatment advice in Tamil. Keep it short and farmer-friendly."""

    print("🌱 Sending prompt to Gemini:", prompt)

    try:
        model = genai.GenerativeModel(model_name=MODEL_NAME)
        response = model.generate_content(prompt)
        print("🧠 Gemini returned:", response.text)
        return response.text
    except Exception as e:
        print("❌ Gemini API error:", e)
        return f"Gemini API error: {e}"
