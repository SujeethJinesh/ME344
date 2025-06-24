#!/usr/bin/env python3
import requests
import json

# Test ChromaDB API endpoints
base_url = "http://localhost:8000"

print("Testing ChromaDB API endpoints...")
print("=" * 50)

# Test different API versions
endpoints = [
    "/api/v1/heartbeat",
    "/api/v1/collections",
    "/heartbeat",
    "/collections",
    "/api/v1",
    "/"
]

for endpoint in endpoints:
    try:
        url = base_url + endpoint
        response = requests.get(url, timeout=2)
        print(f"\n{endpoint}:")
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            print(f"  Response: {response.text[:100]}...")
        else:
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"\n{endpoint}:")
        print(f"  Error: {str(e)}")

# Test with Python client
print("\n" + "=" * 50)
print("Testing with Python ChromaDB client...")
try:
    import chromadb
    client = chromadb.HttpClient(host='localhost', port=8000)
    
    # Test heartbeat
    heartbeat = client.heartbeat()
    print(f"✓ Heartbeat: {heartbeat}")
    
    # List collections
    collections = client.list_collections()
    print(f"✓ Collections: {[c.name for c in collections]}")
    
except Exception as e:
    print(f"✗ Error: {e}")