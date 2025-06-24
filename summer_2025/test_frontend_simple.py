#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test to verify React frontend functionality
"""

import requests
import time
import json

def test_services():
    """Test all backend services"""
    print("Testing backend services...")
    
    services = {
        "ChromaDB": "http://localhost:8000/api/v2/heartbeat",
        "Ollama": "http://localhost:11434",
        "React": "http://localhost:3000"
    }
    
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"OK {name} is running")
            else:
                print(f"WARN {name} returned status {response.status_code}")
        except Exception as e:
            print(f"ERROR {name} is not accessible: {e}")

def test_rag_pipeline():
    """Test RAG pipeline directly"""
    print("\nTesting RAG pipeline...")
    
    try:
        # Test ChromaDB collection access
        from chromadb import ChromaClient
        from chromadb.utils import embedding_functions
        
        client = ChromaClient(host="localhost", port=8000)
        
        # Test embedding function
        ollama_ef = embedding_functions.OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text",
        )
        
        # Get collection
        collection = client.get_collection(
            name="llm_rag_collection",
            embedding_function=ollama_ef
        )
        
        # Test query
        results = collection.query(
            query_texts=["lit"],
            n_results=2
        )
        
        print(f"OK RAG pipeline test successful!")
        print(f"Found {len(results['documents'][0])} documents")
        if results['documents'][0]:
            print(f"Sample: {results['documents'][0][0][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"ERROR RAG pipeline test failed: {e}")
        return False

def test_ollama_generation():
    """Test Ollama generation"""
    print("\nTesting Ollama generation...")
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.1",
                "prompt": "What does 'lit' mean in slang?",
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"OK Ollama generation successful!")
            print(f"Response: {result.get('response', '')[:100]}...")
            return True
        else:
            print(f"ERROR Ollama returned status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"ERROR Ollama generation test failed: {e}")
        return False

if __name__ == "__main__":
    print("Frontend Functionality Test")
    print("=" * 50)
    
    test_services()
    rag_ok = test_rag_pipeline()
    ollama_ok = test_ollama_generation()
    
    print("\nSummary:")
    print(f"RAG Pipeline: {'OK Working' if rag_ok else 'ERROR Failed'}")
    print(f"Ollama Generation: {'OK Working' if ollama_ok else 'ERROR Failed'}")
    
    if rag_ok and ollama_ok:
        print("\nAll backend services working! Frontend issue is in React UI logic.")
    else:
        print("\nBackend issues detected - fix these first.")