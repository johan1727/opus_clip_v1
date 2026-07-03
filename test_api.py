"""Test de conectividad de la Gemini API key configurada en .env."""
import os

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("GEMINI_API_KEY no está configurada. Copia .env.example a .env y agrega tu key.")

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content("Say 'OK' in one word")
    print(f"✅ API Key: WORKING - Response: {response.text.strip()[:20]}")
except Exception as e:
    print(f"❌ API Key: FAILED - {str(e)[:200]}")
