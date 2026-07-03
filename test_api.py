"""Test de conectividad de todas las Gemini API keys configuradas en .env."""
import os

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

keys_raw = os.getenv("GEMINI_API_KEYS", "")
api_keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
if not api_keys:
    single = os.getenv("GEMINI_API_KEY", "").strip()
    if single:
        api_keys = [single]

if not api_keys:
    raise SystemExit("GEMINI_API_KEYS/GEMINI_API_KEY no está configurada. Copia .env.example a .env y agrega tus keys.")

for i, key in enumerate(api_keys, 1):
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content("Say 'OK' in one word")
        print(f"✅ API Key {i}/{len(api_keys)}: WORKING - Response: {response.text.strip()[:20]}")
    except Exception as e:
        print(f"❌ API Key {i}/{len(api_keys)}: FAILED - {str(e)[:100]}")
