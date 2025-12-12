from groq import Groq
import os
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = os.getenv("GROQ_MODEL","llama-3.1-8b-instant")
prompt = "Say hello."
try:
    completion = client.chat.completions.create(model=model, messages=[{"role":"user","content":prompt}], temperature=0.0)
    print("OK:", completion)
except Exception as e:
    print("Groq call failed:", e)
