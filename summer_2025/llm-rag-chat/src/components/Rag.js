import { useEffect } from 'react';
import { ChromaClient, OllamaEmbeddingFunction } from 'chromadb';

const Rag = ({ query, onAugmentedQuery, onProcessStep }) => {
  const fetchAugmentedQuery = async () => {
    const startTime = Date.now();
    
    try {
      // Step 1: Configuration with enhanced educational context
      onProcessStep && onProcessStep("🔧 CONFIGURATION PHASE: Initializing RAG system...", "config");
      onProcessStep && onProcessStep("📚 Educational Note: RAG requires vector DB + embeddings + LLM", "info");
      const chromaUrl = process.env.REACT_APP_CHROMA_URL || 'http://localhost:8000';
      const ollamaUrl = process.env.REACT_APP_OLLAMA_URL || 'http://localhost:11434';
      const embeddingModel = process.env.REACT_APP_EMBEDDING_MODEL || 'nomic-embed-text';
      
      onProcessStep && onProcessStep(`📍 ChromaDB (Vector Store): ${chromaUrl}`, "config");
      onProcessStep && onProcessStep(`📍 Ollama (LLM + Embeddings): ${ollamaUrl}`, "config");
      onProcessStep && onProcessStep(`📍 Embedding Model: ${embeddingModel}`, "config");
      onProcessStep && onProcessStep("📚 Educational Note: Vector databases store semantic embeddings", "info");

      // Step 2: ChromaDB Connection
      onProcessStep && onProcessStep("🗄️ DATABASE PHASE: Connecting to ChromaDB vector database...", "database");
      onProcessStep && onProcessStep("📚 Educational Note: ChromaDB stores 628k+ slang definitions as vectors", "info");
      const client = new ChromaClient({
        path: chromaUrl,
      });

      // Test connection first
      try {
        await client.heartbeat();
        onProcessStep && onProcessStep("✅ ChromaDB connection established", "success");
        onProcessStep && onProcessStep("📚 Educational Note: Heartbeat confirms database is responsive", "info");
      } catch (connectionError) {
        onProcessStep && onProcessStep("❌ ChromaDB connection failed", "error");
        throw new Error(`Failed to connect to ChromaDB at ${chromaUrl}. Make sure ChromaDB is running.`);
      }

      // Step 3: Collection Discovery
      onProcessStep && onProcessStep("📚 Discovering available collections...", "info");
      const collections = await client.listCollections();
      onProcessStep && onProcessStep(`📋 Found ${collections.length} collection(s): ${collections.map(c => c.name).join(', ')}`, "success");

      // Step 4: Embedding Function Setup
      onProcessStep && onProcessStep("🧮 EMBEDDING PHASE: Setting up embedding function...", "embedding");
      onProcessStep && onProcessStep("📚 Educational Note: Embeddings convert text to numerical vectors", "info");
      const embedder = new OllamaEmbeddingFunction({
        url: `${ollamaUrl}/api/embeddings`,
        model: embeddingModel,
      });
      onProcessStep && onProcessStep("✅ Embedding function configured with nomic-embed-text", "success");
      onProcessStep && onProcessStep("📚 Educational Note: This model creates 768-dimensional vectors", "info");

      // Step 5: Collection Retrieval
      onProcessStep && onProcessStep("🎯 Accessing 'llm_rag_collection'...", "info");
      let collection;
      try {
        collection = await client.getCollection({
          name: 'llm_rag_collection',
          embeddingFunction: embedder,
        });
        onProcessStep && onProcessStep("✅ Collection loaded successfully", "success");
      } catch (collectionError) {
        onProcessStep && onProcessStep("❌ Collection 'llm_rag_collection' not found", "error");
        throw new Error(`Collection 'llm_rag_collection' not found. Please run the RAG notebook first to create the collection.`);
      }

      // Step 6: Query Embedding and Search
      onProcessStep && onProcessStep(`🔍 SEARCH PHASE: Searching for relevant documents...`, "search");
      onProcessStep && onProcessStep(`📝 Query: "${query}"`, "query");
      onProcessStep && onProcessStep("⚡ Converting query to vector embedding...", "vector");
      onProcessStep && onProcessStep("📚 Educational Note: Query embedding enables semantic similarity search", "info");
      
      const queryPromise = collection.query({
        queryTexts: [query],
        nResults: 2,
      });

      const timeoutPromise = new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Query timeout after 10 seconds')), 10000)
      );

      const results = await Promise.race([queryPromise, timeoutPromise]);
      
      const searchTime = Date.now() - startTime;
      onProcessStep && onProcessStep(`⚡ Vector search completed in ${searchTime}ms`, "timing");

      // Step 7: Results Processing
      let augmentedQuery = query;
      let context = '';

      if (results && results.documents && results.documents.length > 0) {
        const docCount = results.documents.flat().length;
        onProcessStep && onProcessStep(`📄 RETRIEVAL SUCCESS: Found ${docCount} relevant document(s)`, "success");
        
        // Show similarity scores if available
        if (results.distances && results.distances.length > 0) {
          const scores = results.distances[0].map(d => (1 - d).toFixed(3));
          onProcessStep && onProcessStep(`🎯 Similarity scores: [${scores.join(', ')}]`, "similarity");
          onProcessStep && onProcessStep("📚 Educational Note: Higher scores = more semantically similar", "info");
        }
        
        // Concatenate the retrieved documents to create context
        context = results.documents.flat().join(' ');
        const contextLength = context.length;
        onProcessStep && onProcessStep(`📝 CONTEXT ASSEMBLY: ${contextLength} characters assembled`, "success");
        onProcessStep && onProcessStep("📚 Educational Note: Context provides relevant background to LLM", "info");
        
        // Show preview of retrieved context
        const contextPreview = context.substring(0, 150) + (context.length > 150 ? '...' : '');
        onProcessStep && onProcessStep(`📖 Context preview: "${contextPreview}"`, "context");
        
        // Augment the user's query with the retrieved context
        augmentedQuery = `Use the following information to answer the question:\n\n${context}\n\nQuestion: ${query}`;
        onProcessStep && onProcessStep("🔗 AUGMENTATION PHASE: Query enhanced with context", "success");
        onProcessStep && onProcessStep("📚 Educational Note: This is the 'Augmented' part of RAG", "info");
      } else {
        onProcessStep && onProcessStep("⚠️ No relevant documents found - using original query", "warning");
        console.warn('No relevant documents found for query:', query);
      }

      // Step 8: Final Summary
      const totalTime = Date.now() - startTime;
      onProcessStep && onProcessStep(`🎉 RAG process completed in ${totalTime}ms`, "timing");
      onProcessStep && onProcessStep("🚀 Sending augmented query to LLM...", "info");

      // Pass the augmented query and context back to the parent component
      onAugmentedQuery(augmentedQuery, context);
    } catch (error) {
      console.error('Error querying ChromaDB:', error);
      
      onProcessStep && onProcessStep("❌ RAG process failed", "error");
      
      // Provide specific error messages
      let errorMessage = 'Error querying the knowledge base. ';
      if (error.message.includes('ChromaDB')) {
        errorMessage += 'Please ensure ChromaDB is running.';
        onProcessStep && onProcessStep("💡 Solution: Start ChromaDB with 'chroma run --host localhost --port 8000'", "help");
      } else if (error.message.includes('Collection')) {
        errorMessage += 'Please run the RAG setup notebook first.';
        onProcessStep && onProcessStep("💡 Solution: Run rag.ipynb notebook to create the vector database", "help");
      } else if (error.message.includes('timeout')) {
        errorMessage += 'The query took too long to process.';
        onProcessStep && onProcessStep("💡 Solution: Try a shorter query or check your network connection", "help");
      } else {
        errorMessage += 'Please check your connection and try again.';
        onProcessStep && onProcessStep(`💡 Error details: ${error.message}`, "help");
      }
      
      onProcessStep && onProcessStep(`❌ ${errorMessage}`, "error");
      onAugmentedQuery(query, errorMessage); // Fallback to original query with error context
    }
  };

  useEffect(() => {
    fetchAugmentedQuery();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query]);

  return null; // This component doesn't render anything
};

export default Rag;
