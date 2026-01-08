#!/usr/bin/env python
"""Test fraud detection API"""

import requests
import json

# Sample transactions
transactions = [
    {
        'sender': '0x0000000000000000000000000000000000000001',
        'receiver': '0x0000000000000000000000000000000000000002',
        'value': 10.5,
        'gas_used': 21000,
        'timestamp': 1704067200
    },
    {
        'sender': '0x0000000000000000000000000000000000000003',
        'receiver': '0x0000000000000000000000000000000000000004',
        'value': 5.2,
        'gas_used': 21000,
        'timestamp': 1704067260
    }
]

# Call API
try:
    print("Testing Fraud Detection API...")
    print("=" * 60)
    
    response = requests.post(
        'http://127.0.0.1:5000/api/fraud-detection/analyze',
        json={'transactions': transactions},
        timeout=10
    )
    
    print(f'Status Code: {response.status_code}')
    print(f'Response:')
    print(json.dumps(response.json(), indent=2))
    
except Exception as e:
    print(f'Error: {str(e)}')
    print("\nMake sure the backend is running:")
    print("  python simple_api.py")
