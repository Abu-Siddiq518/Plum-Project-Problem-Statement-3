"""
Test Examples for Medical Report Simplifier API
Run this script to test the API endpoints
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test the health check endpoint"""
    print("\n" + "="*50)
    print("Testing Health Check Endpoint")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_text_analysis_valid():
    """Test text analysis with valid input"""
    print("\n" + "="*50)
    print("Testing Text Analysis - Valid Input")
    print("="*50)
    
    data = {
        "text": "CBC: Hemoglobin 10.2 g/dL (Low), WBC 11,200 /uL (High)"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/analyze/text",
        headers={"Content-Type": "application/json"},
        json=data
    )
    
    print(f"Input: {data['text']}")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_text_analysis_typos():
    """Test text analysis with OCR-like typos"""
    print("\n" + "="*50)
    print("Testing Text Analysis - With Typos")
    print("="*50)
    
    data = {
        "text": "CBC: Hemglobin 10.2 g/dL (Low), WBC 11200 /uL (Hgh)"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/analyze/text",
        headers={"Content-Type": "application/json"},
        json=data
    )
    
    print(f"Input: {data['text']}")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_text_analysis_multiple_tests():
    """Test text analysis with multiple tests"""
    print("\n" + "="*50)
    print("Testing Text Analysis - Multiple Tests")
    print("="*50)
    
    data = {
        "text": """
        CBC Report:
        Hemoglobin 10.2 g/dL (Low)
        WBC 11,200 /uL (High)
        RBC 5.0 million/uL (Normal)
        Glucose 110 mg/dL (High)
        """
    }
    
    response = requests.post(
        f"{BASE_URL}/api/analyze/text",
        headers={"Content-Type": "application/json"},
        json=data
    )
    
    print(f"Input: {data['text']}")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_text_analysis_empty():
    """Test text analysis with empty input"""
    print("\n" + "="*50)
    print("Testing Text Analysis - Empty Input (Error Case)")
    print("="*50)
    
    data = {
        "text": ""
    }
    
    response = requests.post(
        f"{BASE_URL}/api/analyze/text",
        headers={"Content-Type": "application/json"},
        json=data
    )
    
    print(f"Input: {data['text']}")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 400

def test_text_analysis_invalid():
    """Test text analysis with invalid input"""
    print("\n" + "="*50)
    print("Testing Text Analysis - Invalid Input (Error Case)")
    print("="*50)
    
    data = {
        "text": "This is just random text with no medical data"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/analyze/text",
        headers={"Content-Type": "application/json"},
        json=data
    )
    
    print(f"Input: {data['text']}")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 400

def run_all_tests():
    """Run all test cases"""
    print("\n" + "="*70)
    print(" MEDICAL REPORT SIMPLIFIER - API TEST SUITE")
    print("="*70)
    print(f"Testing API at: {BASE_URL}")
    
    tests = [
        ("Health Check", test_health_check),
        ("Text Analysis - Valid Input", test_text_analysis_valid),
        ("Text Analysis - With Typos", test_text_analysis_typos),
        ("Text Analysis - Multiple Tests", test_text_analysis_multiple_tests),
        ("Text Analysis - Empty Input", test_text_analysis_empty),
        ("Text Analysis - Invalid Input", test_text_analysis_invalid),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"\nError in {test_name}: {str(e)}")
            results.append((test_name, "ERROR"))
    
    # Print summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    for test_name, result in results:
        status_symbol = "✓" if result == "PASS" else "✗"
        print(f"{status_symbol} {test_name}: {result}")
    
    passed = sum(1 for _, r in results if r == "PASS")
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*70)

if __name__ == "__main__":
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to the server.")
        print("Please make sure the Flask app is running on http://localhost:5000")
        print("\nRun the server with: python app.py")