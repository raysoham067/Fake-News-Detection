#!/usr/bin/env python3
"""
Test script for the Fake News Detection application
"""

import requests
import json
import time

def test_fake_news_detection():
    """Test the fake news detection API"""
    
    # Test cases
    test_cases = [
        {
            "text": "Scientists discover that aliens are living among us and controlling our thoughts",
            "expected": "fake"
        },
        {
            "text": "New study shows benefits of regular exercise on mental health",
            "expected": "real"
        },
        {
            "text": "Breaking: Time travel machine invented by 15-year-old in garage",
            "expected": "fake"
        },
        {
            "text": "Research indicates climate change impact on global agriculture",
            "expected": "real"
        }
    ]
    
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Fake News Detection API")
    print("=" * 50)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print("❌ Health check failed")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to the application. Make sure it's running on http://localhost:5000")
        return
    
    # Test detection endpoint
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}:")
        print(f"Text: {test_case['text']}")
        print(f"Expected: {test_case['expected']}")
        
        try:
            response = requests.post(
                f"{base_url}/detect",
                json={"text": test_case['text']},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = "fake" if result['is_fake'] else "real"
                confidence = result['confidence']
                
                print(f"✅ Prediction: {prediction}")
                print(f"   Confidence - Fake: {confidence['fake']}%, Real: {confidence['real']}%")
                
                if prediction == test_case['expected']:
                    print("   🎯 Correct prediction!")
                else:
                    print("   ⚠️  Incorrect prediction")
                    
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
        
        time.sleep(1)  # Small delay between requests
    
    print("\n" + "=" * 50)
    print("🏁 Testing completed!")

def test_invalid_inputs():
    """Test the API with invalid inputs"""
    
    base_url = "http://localhost:5000"
    
    print("\n🧪 Testing Invalid Inputs")
    print("=" * 50)
    
    # Test empty text
    try:
        response = requests.post(
            f"{base_url}/detect",
            json={"text": ""},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 400:
            print("✅ Empty text properly rejected")
        else:
            print(f"❌ Empty text should be rejected, got status {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    
    # Test missing text field
    try:
        response = requests.post(
            f"{base_url}/detect",
            json={},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 400:
            print("✅ Missing text field properly rejected")
        else:
            print(f"❌ Missing text field should be rejected, got status {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    print("🚀 Fake News Detection Test Suite")
    print("Make sure the Flask application is running on http://localhost:5000")
    print()
    
    test_fake_news_detection()
    test_invalid_inputs()
