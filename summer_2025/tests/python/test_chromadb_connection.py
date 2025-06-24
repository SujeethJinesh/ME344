#!/usr/bin/env python3
import chromadb

# Connect to ChromaDB server
client = chromadb.HttpClient(host='localhost', port=8000)

# Test connection
try:
    print("Testing ChromaDB connection...")
    heartbeat = client.heartbeat()
    print(f"✓ Connection successful: {heartbeat}")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    exit(1)

# List collections
try:
    collections = client.list_collections()
    print(f"\nFound {len(collections)} collections:")
    for col in collections:
        print(f"  - {col.name}")
    
    if len(collections) == 0:
        print("\n⚠️  No collections found. You need to run the rag.ipynb notebook first!")
except Exception as e:
    print(f"✗ Failed to list collections: {e}")