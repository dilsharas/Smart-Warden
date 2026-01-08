#!/usr/bin/env python3
"""
CORS Testing Script - Verify backend API CORS configuration
"""

import requests
import json
from colorama import Fore, Style, init

init(autoreset=True)

def test_cors():
    """Test CORS configuration"""
    
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}CORS Configuration Test")
    print(f"{Fore.CYAN}{'='*60}\n")
    
    api_url = "http://localhost:5000"
    frontend_origins = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8501",
        "http://localhost:3000"
    ]
    
    # Test 1: Check if API is running
    print(f"{Fore.YELLOW}[1/4] Checking if API is running...")
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"{Fore.GREEN}✓ API is running")
            print(f"  Status: {response.json()['status']}")
        else:
            print(f"{Fore.RED}✗ API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"{Fore.RED}✗ Cannot connect to API at {api_url}")
        print(f"  Make sure backend is running: python simple_api.py")
        return False
    except Exception as e:
        print(f"{Fore.RED}✗ Error: {e}")
        return False
    
    # Test 2: Check CORS headers
    print(f"\n{Fore.YELLOW}[2/4] Checking CORS headers...")
    try:
        headers = {
            "Origin": "http://localhost:8000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        }
        response = requests.options(f"{api_url}/api/analyze", headers=headers, timeout=5)
        
        cors_headers = {
            'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
            'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
            'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers'),
        }
        
        if cors_headers['Access-Control-Allow-Origin']:
            print(f"{Fore.GREEN}✓ CORS headers present")
            for key, value in cors_headers.items():
                if value:
                    print(f"  {key}: {value}")
        else:
            print(f"{Fore.RED}✗ CORS headers missing")
            return False
    except Exception as e:
        print(f"{Fore.RED}✗ Error checking CORS: {e}")
        return False
    
    # Test 3: Test analysis endpoint
    print(f"\n{Fore.YELLOW}[3/4] Testing analysis endpoint...")
    try:
        test_contract = """pragma solidity ^0.8.0;
contract Test {
    function test() public {}
}"""
        
        payload = {
            "contract_code": test_contract,
            "aiAnalysis": True,
            "patternAnalysis": True,
            "externalTools": False,
            "modelType": "binary",
            "confidenceThreshold": 50,
            "timeout": 30,
            "reportFormat": "detailed",
            "includeRecommendations": True,
            "parallelAnalysis": True
        }
        
        response = requests.post(
            f"{api_url}/api/analyze",
            json=payload,
            headers={"Origin": "http://localhost:8000"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"{Fore.GREEN}✓ Analysis endpoint working")
            print(f"  Risk Score: {result.get('risk_score', 'N/A')}")
            print(f"  Vulnerabilities: {len(result.get('vulnerabilities', []))}")
        else:
            print(f"{Fore.RED}✗ Analysis failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"{Fore.RED}✗ Error testing analysis: {e}")
        return False
    
    # Test 4: Test all allowed origins
    print(f"\n{Fore.YELLOW}[4/4] Testing allowed origins...")
    all_allowed = True
    for origin in frontend_origins:
        try:
            headers = {"Origin": origin}
            response = requests.get(f"{api_url}/health", headers=headers, timeout=5)
            allowed_origin = response.headers.get('Access-Control-Allow-Origin')
            
            if allowed_origin:
                print(f"{Fore.GREEN}✓ {origin}")
            else:
                print(f"{Fore.YELLOW}⚠ {origin} (no CORS header)")
                all_allowed = False
        except Exception as e:
            print(f"{Fore.RED}✗ {origin} - Error: {e}")
            all_allowed = False
    
    # Summary
    print(f"\n{Fore.CYAN}{'='*60}")
    if all_allowed:
        print(f"{Fore.GREEN}✓ All tests passed! CORS is properly configured.")
    else:
        print(f"{Fore.YELLOW}⚠ Some tests had warnings. Check configuration.")
    print(f"{Fore.CYAN}{'='*60}\n")
    
    return True

if __name__ == "__main__":
    try:
        success = test_cors()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n{Fore.RED}Unexpected error: {e}")
        exit(1)
