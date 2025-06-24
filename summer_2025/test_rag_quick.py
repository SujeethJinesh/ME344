#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test to verify RAG pipeline with current setup
"""

import requests
import json

def test_chromadb():
    """Test ChromaDB connection and collection"""
    print("Testing ChromaDB...")
    try:
        # Test heartbeat
        response = requests.get("http://localhost:8000/api/v2/heartbeat", timeout=5)
        if response.status_code == 200:
            print("  OK ChromaDB is responding")
        else:
            print(f"  ERROR ChromaDB returned {response.status_code}")
            return False
            
        # Test collections
        response = requests.get("http://localhost:8000/api/v2/collections", timeout=5)
        print(f"  Collections endpoint status: {response.status_code}")
        
        return True
    except Exception as e:
        print(f"  ERROR ChromaDB test failed: {e}")
        return False

def test_ollama():
    """Test Ollama embedding"""
    print("Testing Ollama embeddings...")
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": "test query"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            embedding = result.get('embedding', [])
            print(f"  OK Ollama embedding generated: {len(embedding)} dimensions")
            return True
        else:
            print(f"  ERROR Ollama returned {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ERROR Ollama test failed: {e}")
        return False

def test_react_app():
    """Test React app is serving"""
    print("Testing React app...")
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print("  OK React app is serving")
            return True
        else:
            print(f"  ERROR React app returned {response.status_code}")
            return False
    except Exception as e:
        print(f"  ERROR React app test failed: {e}")
        return False

if __name__ == "__main__":
    print("Quick RAG Pipeline Test")
    print("=" * 30)
    
    chromadb_ok = test_chromadb()
    ollama_ok = test_ollama()
    react_ok = test_react_app()
    
    print("\nSummary:")
    print(f"ChromaDB: {'OK' if chromadb_ok else 'FAIL'}")
    print(f"Ollama: {'OK' if ollama_ok else 'FAIL'}")
    print(f"React: {'OK' if react_ok else 'FAIL'}")
    
    if chromadb_ok and ollama_ok and react_ok:
        print("\nAll services are working! Try the frontend now.")
    else:
        print("\nSome services failed. Check the logs.")