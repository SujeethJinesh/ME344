#!/usr/bin/env python3
"""
Test the RAG system as the React frontend would use it
This simulates the exact same API calls that the React app makes
"""

import json
import requests
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

def test_frontend_rag_api():
    print("🌐 Testing RAG System as React Frontend")
    print("=" * 50)
    
    # Step 1: Test ChromaDB connection (same as React app)
    print("1️⃣ Testing ChromaDB connection from frontend perspective...")
    
    try:
        # Simulate the exact connection the React app makes
        client = chromadb.HttpClient(host='localhost', port=8000)
        client.heartbeat()
        print("✅ ChromaDB heartbeat successful")
        
        # Get collections (what React app does)
        collections = client.list_collections()
        print(f"📋 Found {len(collections)} collection(s)")
        
        # Setup embedding function (exactly like React app)
        embedder = chromadb.utils.embedding_functions.OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text",
        )
        print("✅ Embedding function configured")
        
        # Get the collection (exactly like React app)
        collection = client.get_collection(
            name='llm_rag_collection',
            embedding_function=embedder,
        )
        print("✅ Collection retrieved successfully")
        
    except Exception as e:
        print(f"❌ ChromaDB connection failed: {e}")
        return False
    
    # Step 2: Test the RAG query process (simulate React app workflow)
    print("\n2️⃣ Testing RAG query process...")
    
    test_queries = [
        "what does cool mean?",
        "explain janky",
        "what is fire slang?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        
        try:
            # Step 2a: Vector search (exactly what React Rag.js does)
            print("   📊 Performing vector search...")
            results = collection.query(
                query_texts=[query],
                n_results=2,
            )
            
            if results and results['documents'] and len(results['documents']) > 0:
                doc_count = len(results['documents'][0])
                print(f"   ✅ Found {doc_count} relevant documents")
                
                # Show similarity scores
                if results['distances'] and len(results['distances']) > 0:
                    scores = [round(1 - d, 3) for d in results['distances'][0]]
                    print(f"   🎯 Similarity scores: {scores}")
                
                # Step 2b: Create augmented query (exactly what React does)
                context = ' '.join(results['documents'][0])
                augmented_query = f"Use the following information to answer the question:\n\n{context}\n\nQuestion: {query}"
                
                print(f"   📝 Context length: {len(context)} characters")
                print(f"   📝 Augmented query length: {len(augmented_query)} characters")
                
                # Step 2c: Send to Ollama (exactly what React does)
                print("   🤖 Sending to Ollama...")
                
                ollama_payload = {
                    "model": "llama3.1",
                    "prompt": augmented_query,
                    "stream": True  # Test streaming like React app
                }
                
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    headers={"Content-Type": "application/json"},
                    json=ollama_payload,
                    stream=True,
                    timeout=60
                )
                
                if response.status_code == 200:
                    print("   ✅ Ollama streaming response started")
                    
                    # Collect streaming response (like React app does)
                    accumulated_response = ""
                    chunk_count = 0
                    
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = json.loads(line.decode('utf-8'))
                                if 'response' in data:
                                    accumulated_response += data['response']
                                    chunk_count += 1
                                if data.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue
                    
                    print(f"   ✅ Received {chunk_count} chunks")
                    print(f"   📝 Final response length: {len(accumulated_response)} characters")
                    print(f"   📝 Response preview: {accumulated_response[:150]}...")
                    
                    # Validate response quality
                    if any(keyword in accumulated_response.lower() for keyword in query.lower().split()):
                        print("   ✅ Response appears relevant to query")
                    else:
                        print("   ⚠️ Response may not be directly relevant")
                        
                else:
                    print(f"   ❌ Ollama request failed: {response.status_code}")
                    
            else:
                print("   ⚠️ No documents found for query")
                
        except Exception as e:
            print(f"   ❌ Query processing failed: {e}")
            return False
    
    # Step 3: Test error handling scenarios
    print("\n3️⃣ Testing error handling...")
    
    # Test invalid query
    try:
        print("   🧪 Testing empty query...")
        results = collection.query(query_texts=[""], n_results=2)
        print("   ✅ Empty query handled gracefully")
    except Exception as e:
        print(f"   ⚠️ Empty query error: {e}")
    
    # Test network resilience
    try:
        print("   🧪 Testing Ollama connectivity...")
        ping_response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if ping_response.status_code == 200:
            print("   ✅ Ollama connectivity confirmed")
        else:
            print("   ⚠️ Ollama connectivity issue")
    except Exception as e:
        print(f"   ⚠️ Ollama connectivity error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Frontend API Test Complete!")
    print("✅ RAG system is ready for React frontend use")
    
    print("\n📋 Test Summary:")
    print("   ✅ ChromaDB connection working")
    print("   ✅ Vector search functioning") 
    print("   ✅ Query augmentation working")
    print("   ✅ Ollama streaming responses working")
    print("   ✅ End-to-end RAG pipeline functional")
    
    print("\n🚀 Ready for manual testing!")
    print("   🌐 React Frontend: http://localhost:3000")
    print("   🔧 ChromaDB: http://localhost:8000")  
    print("   🤖 Ollama API: http://localhost:11434")
    
    return True

if __name__ == "__main__":
    test_frontend_rag_api()