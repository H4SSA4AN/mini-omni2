#!/usr/bin/env python3
"""
Test script for the reverse proxy functionality
"""

import requests
import json
import time

def test_proxy():
    base_url = "http://localhost:60808"
    
    print("Testing reverse proxy functionality...")
    
    # Test 1: Health check
    print("\n1. Testing proxy health check...")
    try:
        resp = requests.get(f"{base_url}/proxy/health")
        print(f"   Status: {resp.status_code}")
        print(f"   Response: {resp.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: Connection check
    print("\n2. Testing proxy connection check...")
    try:
        resp = requests.post(f"{base_url}/proxy/check", 
                           json={"url": "http://httpbin.org/status/200"})
        print(f"   Status: {resp.status_code}")
        data = resp.json()
        print(f"   Response: {data}")
        if data.get('ok'):
            print(f"   ✅ Health check successful: {data.get('message', 'N/A')}")
        else:
            print(f"   ❌ Health check failed: {data.get('message', 'N/A')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: Proxy GET request
    print("\n3. Testing proxy GET request...")
    try:
        target_url = "http://httpbin.org/get"
        proxy_url = f"{base_url}/api/get?target_url={requests.utils.quote(target_url)}"
        resp = requests.get(proxy_url)
        print(f"   Status: {resp.status_code}")
        data = resp.json()
        print(f"   Response URL: {data.get('url', 'N/A')}")
        # Verify the target URL was correctly constructed
        expected_url = f"{target_url}/get"
        actual_url = data.get('url', '')
        if actual_url == expected_url:
            print(f"   ✅ Path correctly stripped: {actual_url}")
        else:
            print(f"   ❌ Path not stripped correctly. Expected: {expected_url}, Got: {actual_url}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 4: Proxy POST request
    print("\n4. Testing proxy POST request...")
    try:
        target_url = "http://httpbin.org/post"
        proxy_url = f"{base_url}/api/post?target_url={requests.utils.quote(target_url)}"
        resp = requests.post(proxy_url, json={"test": "data"})
        print(f"   Status: {resp.status_code}")
        data = resp.json()
        print(f"   Response JSON: {data.get('json', 'N/A')}")
        # Verify the target URL was correctly constructed
        expected_url = f"{target_url}/post"
        actual_url = data.get('url', '')
        if actual_url == expected_url:
            print(f"   ✅ Path correctly stripped: {actual_url}")
        else:
            print(f"   ❌ Path not stripped correctly. Expected: {expected_url}, Got: {actual_url}")
    except Exception as e:
        print(f"   Error: {e}")
    
        # Test 5: Proxy with header
    print("\n5. Testing proxy with X-Target-URL header...")
    try:
        target_url = "http://httpbin.org/headers"
        resp = requests.get(f"{base_url}/api/headers", 
                           headers={"X-Target-URL": target_url})
        print(f"   Status: {resp.status_code}")
        data = resp.json()
        print(f"   Headers received: {len(data.get('headers', {}))}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 6: Test health endpoint specifically
    print("\n6. Testing health endpoint...")
    try:
        target_url = "http://httpbin.org/health"
        proxy_url = f"{base_url}/api/health?target_url={requests.utils.quote(target_url)}"
        resp = requests.get(proxy_url)
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"   ✅ Health endpoint works correctly")
        else:
            print(f"   ❌ Health endpoint failed: {resp.status_code}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\nProxy test completed!")

if __name__ == "__main__":
    test_proxy()
