#!/usr/bin/env python3
"""
Test script to check ChromaDB connection and data
"""

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

def test_chromadb():
    print("🔧 Testing ChromaDB connection...")
    
    try:
        # Connect to ChromaDB
        client = chromadb.HttpClient(host='localhost', port=8000)
        
        # Test heartbeat
        client.heartbeat()
        print("✅ ChromaDB connection successful")
        
        # List collections
        collections = client.list_collections()
        print(f"📋 Found {len(collections)} collections:")
        for collection in collections:
            print(f"  - {collection.name}")
        
        # Check for the RAG collection specifically
        try:
            embedder = embedding_functions.OllamaEmbeddingFunction(
                url="http://localhost:11434/api/embeddings",
                model_name="nomic-embed-text",
            )
            
            collection = client.get_collection(
                name='llm_rag_collection',
                embedding_function=embedder,
            )
            
            # Get collection info
            count = collection.count()
            print(f"✅ Found 'llm_rag_collection' with {count} documents")
            
            if count > 0:
                # Test a simple query
                print("🔍 Testing vector search...")
                results = collection.query(
                    query_texts=["what is cool"],
                    n_results=2,
                )
                
                if results and results['documents']:
                    print(f"🎯 Query returned {len(results['documents'][0])} results:")
                    for i, doc in enumerate(results['documents'][0]):
                        distance = results['distances'][0][i] if results['distances'] else 'N/A'
                        print(f"  {i+1}. (similarity: {1-distance:.3f}): {doc[:100]}...")
                else:
                    print("⚠️ Query returned no results")
            else:
                print("📊 Collection is empty - need to populate with data first")
                
        except Exception as e:
            print(f"❌ Collection 'llm_rag_collection' not found: {e}")
            print("💡 Need to run the RAG notebook first to create and populate the collection")
        
    except Exception as e:
        print(f"❌ ChromaDB connection failed: {e}")
        print("💡 Make sure ChromaDB is running: chroma run --host localhost --port 8000 --path ./chroma")

if __name__ == "__main__":
    test_chromadb()