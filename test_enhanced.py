#!/usr/bin/env python3
"""
Test script for the Enhanced Fake News Detection application
Tests that the model doesn't give 50/50 predictions
"""

import requests
import json
import time

def test_enhanced_model():
    """Test the enhanced fake news detection model"""
    
    # Test cases with expected outcomes
    test_cases = [
        {
            "text": "BREAKING: Scientists discover that aliens are living among us and controlling our thoughts",
            "expected": "fake",
            "description": "Obvious fake news with sensational claims"
        },
        {
            "text": "New study shows benefits of regular exercise on mental health",
            "expected": "real",
            "description": "Plausible scientific study"
        },
        {
            "text": "SHOCKING: Time travel machine invented by 15-year-old in garage",
            "expected": "fake",
            "description": "Clickbait fake news"
        },
        {
            "text": "Research indicates climate change impact on global agriculture",
            "expected": "real",
            "description": "Real scientific research"
        },
        {
            "text": "The moon landing was completely faked by Hollywood",
            "expected": "fake",
            "description": "Conspiracy theory"
        },
        {
            "text": "Apple releases new iPhone with improved camera system",
            "expected": "real",
            "description": "Real technology news"
        }
    ]
    
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Enhanced Fake News Detection Model")
    print("=" * 60)
    print("Testing that the model doesn't give 50/50 predictions...")
    print()
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed")
            print(f"   Model loaded: {health_data.get('model_loaded', False)}")
            if 'training_data_size' in health_data:
                print(f"   Training data size: {health_data['training_data_size']}")
        else:
            print("❌ Health check failed")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to the application. Make sure it's running on http://localhost:5000")
        return
    
    # Test detection endpoint
    total_tests = 0
    passed_tests = 0
    confidence_issues = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case['description']}")
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
                fake_conf = confidence['fake']
                real_conf = confidence['real']
                
                print(f"✅ Prediction: {prediction}")
                print(f"   Confidence - Fake: {fake_conf}%, Real: {real_conf}%")
                
                # Check for 50/50 predictions
                confidence_diff = abs(fake_conf - real_conf)
                if confidence_diff < 10:
                    print(f"   ⚠️  WARNING: Low confidence difference ({confidence_diff}%) - possible 50/50 prediction")
                    confidence_issues += 1
                elif confidence_diff < 20:
                    print(f"   ⚠️  Moderate confidence difference ({confidence_diff}%)")
                else:
                    print(f"   🎯 Good confidence difference ({confidence_diff}%)")
                
                # Check prediction accuracy
                if prediction == test_case['expected']:
                    print("   🎯 Correct prediction!")
                    passed_tests += 1
                else:
                    print("   ❌ Incorrect prediction")
                
                # Check prediction strength
                if 'prediction_strength' in result:
                    print(f"   💪 Prediction strength: {result['prediction_strength']}")
                
                total_tests += 1
                    
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
        
        time.sleep(1)  # Small delay between requests
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 TESTING SUMMARY")
    print("=" * 60)
    print(f"Total tests: {total_tests}")
    print(f"Passed tests: {passed_tests}")
    print(f"Accuracy: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "N/A")
    print(f"Confidence issues (low diff): {confidence_issues}")
    
    if confidence_issues == 0:
        print("🎉 No 50/50 predictions detected! Model is working well.")
    else:
        print(f"⚠️  {confidence_issues} test(s) had low confidence differences.")
    
    # Test invalid inputs
    print("\n🧪 Testing Input Validation")
    print("-" * 30)
    
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
    
    # Test short text
    try:
        response = requests.post(
            f"{base_url}/detect",
            json={"text": "Hi"},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 400:
            print("✅ Short text properly rejected")
        else:
            print(f"❌ Short text should be rejected, got status {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    print("🚀 Enhanced Fake News Detection Test Suite")
    print("Make sure the Flask application is running on http://localhost:5000")
    print()
    
    test_enhanced_model()
