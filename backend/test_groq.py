from dotenv import load_dotenv
import os
load_dotenv()

from langchain_groq import ChatGroq

api_key = os.getenv("GROQ_API_KEY")
print(f"API Key loaded: {api_key[:20]}..." if api_key else "❌ NO API KEY FOUND")

try:
    llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    response = llm.predict("Say hello")
    print(f"✅ Groq works! Response: {response}")
except Exception as e:
    print(f"❌ Error: {e}")