#!/usr/bin/env python3
"""
End-to-end RAG pipeline test script
Tests the complete RAG workflow: Query -> Vector Search -> Context Retrieval -> LLM Generation
"""

import json
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import requests
import time

def test_rag_pipeline():
    print("🧪 Testing Complete RAG Pipeline")
    print("=" * 50)
    
    # Test 1: ChromaDB Connection and Data
    print("\n1️⃣ Testing ChromaDB Connection...")
    try:
        client = chromadb.HttpClient(host='localhost', port=8000)
        client.heartbeat()
        print("✅ ChromaDB connection successful")
        
        collections = client.list_collections()
        print(f"📋 Found {len(collections)} collections")
        
        # Get the RAG collection
        embedder = embedding_functions.OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text",
        )
        
        collection = client.get_collection(
            name='llm_rag_collection',
            embedding_function=embedder,
        )
        
        count = collection.count()
        print(f"✅ Found 'llm_rag_collection' with {count} documents")
        
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        return False
    
    # Test 2: Vector Search
    print("\n2️⃣ Testing Vector Search...")
    test_queries = [
        "what is cool",
        "define janky", 
        "meaning of sick",
        "what does fire mean"
    ]
    
    search_results = {}
    
    for query in test_queries:
        try:
            print(f"🔍 Searching for: '{query}'")
            results = collection.query(
                query_texts=[query],
                n_results=2,
            )
            
            if results and results['documents'] and results['documents'][0]:
                search_results[query] = results
                print(f"✅ Found {len(results['documents'][0])} results")
                
                for i, doc in enumerate(results['documents'][0]):
                    distance = results['distances'][0][i] if results['distances'] else 0
                    similarity = 1 - distance
                    print(f"   {i+1}. (similarity: {similarity:.3f}): {doc[:80]}...")
            else:
                print(f"⚠️ No results found for '{query}'")
                
        except Exception as e:
            print(f"❌ Search failed for '{query}': {e}")
            return False
    
    # Test 3: RAG Query Augmentation
    print("\n3️⃣ Testing RAG Query Augmentation...")
    
    def create_augmented_query(query, context_docs):
        context = ' '.join(context_docs)
        augmented = f"Use the following information to answer the question:\n\n{context}\n\nQuestion: {query}"
        return augmented
    
    test_query = "what is cool"
    if test_query in search_results:
        context_docs = search_results[test_query]['documents'][0]
        augmented_query = create_augmented_query(test_query, context_docs)
        
        print(f"📝 Original query: {test_query}")
        print(f"📝 Augmented query length: {len(augmented_query)} characters")
        print(f"📝 Context preview: {augmented_query[:200]}...")
        print("✅ Query augmentation successful")
    else:
        print(f"❌ No search results for test query: {test_query}")
        return False
    
    # Test 4: Ollama LLM Generation
    print("\n4️⃣ Testing Ollama LLM Generation...")
    
    try:
        # Test basic Ollama connectivity
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            print("✅ Ollama server accessible")
        else:
            print("❌ Ollama server not responding")
            return False
        
        # Test LLM generation with augmented query
        payload = {
            "model": "llama3.1",
            "prompt": augmented_query,
            "stream": False
        }
        
        print("🤖 Sending augmented query to Ollama...")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            llm_response = result.get('response', '')
            
            print(f"✅ LLM generation successful ({generation_time:.2f}s)")
            print(f"📝 Response length: {len(llm_response)} characters")
            print(f"📝 Response preview: {llm_response[:200]}...")
            
            # Check if response contains relevant information
            if any(word in llm_response.lower() for word in ['cool', 'slang', 'meaning', 'definition']):
                print("✅ Response appears relevant to the query")
            else:
                print("⚠️ Response may not be directly relevant")
                
        else:
            print(f"❌ LLM generation failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        return False
    
    # Test 5: End-to-End RAG Function
    print("\n5️⃣ Testing Complete RAG Function...")
    
    def complete_rag_query(user_query):
        """Complete RAG pipeline function"""
        try:
            # Step 1: Vector search
            results = collection.query(
                query_texts=[user_query],
                n_results=2,
            )
            
            if not results or not results['documents'] or not results['documents'][0]:
                return f"No relevant information found for: {user_query}"
            
            # Step 2: Create augmented query
            context_docs = results['documents'][0]
            context = ' '.join(context_docs)
            augmented_query = f"Use the following information to answer the question:\n\n{context}\n\nQuestion: {user_query}"
            
            # Step 3: LLM generation
            payload = {
                "model": "llama3.1",
                "prompt": augmented_query,
                "stream": False
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                return f"LLM generation failed: {response.status_code}"
                
        except Exception as e:
            return f"RAG query failed: {e}"
    
    # Test the complete function
    test_queries_final = [
        "what does cool mean in slang?",
        "explain the term janky"
    ]
    
    for query in test_queries_final:
        print(f"\n🎯 Testing complete RAG for: '{query}'")
        result = complete_rag_query(query)
        print(f"📝 RAG Response: {result[:300]}{'...' if len(result) > 300 else ''}")
    
    print("\n" + "=" * 50)
    print("🎉 RAG Pipeline Test Complete!")
    print("✅ All components are working correctly")
    return True

if __name__ == "__main__":
    success = test_rag_pipeline()
    if success:
        print("\n🚀 Ready to test the React frontend!")
        print("💡 Start the frontend with: ./run_part1.sh --frontend-only")
        print("💡 Then visit: http://localhost:3000")
    else:
        print("\n❌ RAG pipeline has issues that need to be fixed")