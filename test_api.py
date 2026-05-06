# test_api.py
import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*50)
    print("Testing Health Check...")
    print("="*50)
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n" + "="*50)
    print("Testing Single Prediction...")
    print("="*50)
    
    test_data = {
        "temperature_c": 32.5,
        "humidity_percent": 68,
        "dust_level": 242
    }
    
    print(f"Request Data: {json.dumps(test_data, indent=2)}")
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n" + "="*50)
    print("Testing Batch Prediction...")
    print("="*50)
    
    test_data = {
        "samples": [
            {"temperature_c": 35, "humidity_percent": 65, "dust_level": 350},
            {"temperature_c": 28, "humidity_percent": 55, "dust_level": 120},
            {"temperature_c": 22, "humidity_percent": 45, "dust_level": 80},
            {"temperature_c": 31, "humidity_percent": 75, "dust_level": 400},
            {"temperature_c": 26, "humidity_percent": 60, "dust_level": 200}
        ]
    }
    
    print(f"Request Data: {json.dumps(test_data, indent=2)}")
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\nProcessed {result['total_samples']} samples:")
        for i, pred in enumerate(result['predictions'], 1):
            print(f"\nSample {i}:")
            print(f"  Command: {pred['command']}")
            print(f"  Confidence: {pred['confidence']:.2%}")
            print(f"  Reasoning: {pred['reasoning']}")
    return response.status_code == 200

def test_model_info():
    """Test model info endpoint"""
    print("\n" + "="*50)
    print("Testing Model Info...")
    print("="*50)
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_invalid_data():
    """Test invalid data (should return 422)"""
    print("\n" + "="*50)
    print("Testing Invalid Data (Validation)...")
    print("="*50)
    
    # Invalid: Temperature too high
    invalid_data = {
        "temperature_c": 100,
        "humidity_percent": 68,
        "dust_level": 242
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=invalid_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code} (Expected: 422)")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 422

if __name__ == "__main__":
    print("\n🚀 Testing AC Control API")
    print("="*50)
    print("Make sure the API is running: python app.py")
    print("or: uvicorn app:app --reload")
    print("\nRunning tests...")
    
    tests = [
        ("Health Check", test_health),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Model Info", test_model_info),
        ("Invalid Data Validation", test_invalid_data)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} failed with error: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {test_name}")