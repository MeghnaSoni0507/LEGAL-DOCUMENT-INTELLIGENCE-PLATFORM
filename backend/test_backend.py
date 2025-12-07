"""
Test script to verify backend endpoints are working
Run this to test your backend without the frontend
"""

import requests
import json

BACKEND_URL = "http://127.0.0.1:5000"

def test_home():
    """Test if backend is running"""
    print("\n🧪 Testing Home Endpoint...")
    try:
        response = requests.get(f"{BACKEND_URL}/")
        print(f"✅ Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_ask_ai():
    """Test AI Q&A endpoint"""
    print("\n🧪 Testing Ask AI Endpoint...")
    
    test_data = {
        "question": "What is this document about?",
        "document_text": "This is a legal contract between Party A and Party B. The contract includes termination clauses and penalty fees of $5000."
    }
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/ask-ai",
            headers={"Content-Type": "application/json"},
            json=test_data
        )
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Answer: {data.get('answer', 'No answer')}")
            return True
        else:
            print(f"❌ Error: {response.json()}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_detect_anomalies():
    """Test anomaly detection endpoint"""
    print("\n🧪 Testing Detect Anomalies Endpoint...")
    
    test_data = {
        "document_text": "This contract includes a termination clause. The penalty for breach is $10000. There is a confidentiality agreement that must be maintained."
    }
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/detect-anomalies",
            headers={"Content-Type": "application/json"},
            json=test_data
        )
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {len(data.get('found_clauses', []))} risky clauses")
            print(f"AI Feedback: {data.get('ai_feedback', 'No feedback')[:100]}...")
            return True
        else:
            print(f"❌ Error: {response.json()}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_env():
    """Check if environment is configured"""
    print("\n🧪 Checking Environment Configuration...")
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        print(f"✅ OpenAI API Key found: {api_key[:20]}...")
        return True
    else:
        print("❌ OpenAI API Key NOT found in .env file")
        print("   Please create a .env file with: OPENAI_API_KEY=sk-proj-xxxxx")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Legal Document Intelligence - Backend Test Suite")
    print("=" * 60)
    
    # Check environment first
    env_ok = check_env()
    
    # Test endpoints
    home_ok = test_home()
    
    if home_ok and env_ok:
        print("\n⏳ Testing AI features (this may take a few seconds)...")
        ai_ok = test_ask_ai()
        anomaly_ok = test_detect_anomalies()
        
        print("\n" + "=" * 60)
        print("📊 Test Results Summary:")
        print("=" * 60)
        print(f"Backend Running: {'✅' if home_ok else '❌'}")
        print(f"Environment Config: {'✅' if env_ok else '❌'}")
        print(f"Ask AI: {'✅' if ai_ok else '❌'}")
        print(f"Detect Anomalies: {'✅' if anomaly_ok else '❌'}")
        print("=" * 60)
        
        if all([home_ok, env_ok, ai_ok, anomaly_ok]):
            print("\n🎉 All tests passed! Your backend is working correctly.")
        else:
            print("\n⚠️ Some tests failed. Check the errors above.")
    else:
        print("\n⚠️ Backend is not running or environment is not configured.")
        print("   1. Make sure backend is running: python app.py")
        print("   2. Make sure .env file exists with OPENAI_API_KEY")