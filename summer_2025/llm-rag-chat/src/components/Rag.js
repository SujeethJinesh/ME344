import { useEffect } from 'react';
import { ChromaClient, OllamaEmbeddingFunction } from 'chromadb';

const Rag = ({ query, onAugmentedQuery }) => {
  const fetchAugmentedQuery = async () => {
    try {
      // Use environment variables for configuration
      const chromaUrl = process.env.REACT_APP_CHROMA_URL || 'http://localhost:8000';
      const ollamaUrl = process.env.REACT_APP_OLLAMA_URL || 'http://localhost:11434';
      const embeddingModel = process.env.REACT_APP_EMBEDDING_MODEL || 'nomic-embed-text';

      // Initialize the ChromaDB client
      const client = new ChromaClient({
        path: chromaUrl,
      });

      // Test connection first
      try {
        await client.heartbeat();
      } catch (connectionError) {
        throw new Error(`Failed to connect to ChromaDB at ${chromaUrl}. Make sure ChromaDB is running.`);
      }

      const collections = await client.listCollections();
      console.log('Available collections: ', collections);

      const embedder = new OllamaEmbeddingFunction({
        url: `${ollamaUrl}/api/embeddings`,
        model: embeddingModel,
      });

      // Get the collection with the embedding function
      let collection;
      try {
        collection = await client.getCollection({
          name: 'llm_rag_collection',
          embeddingFunction: embedder,
        });
      } catch (collectionError) {
        throw new Error(`Collection 'llm_rag_collection' not found. Please run the RAG notebook first to create the collection.`);
      }

      // Perform the query with timeout
      const queryPromise = collection.query({
        queryTexts: [query],
        nResults: 2,
      });

      const timeoutPromise = new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Query timeout after 10 seconds')), 10000)
      );

      const results = await Promise.race([queryPromise, timeoutPromise]);

      // Process the results
      let augmentedQuery = query;
      let context = '';

      if (results && results.documents && results.documents.length > 0) {
        // Concatenate the retrieved documents to create context
        context = results.documents.flat().join(' ');
        // Augment the user's query with the retrieved context
        augmentedQuery = `Use the following information to answer the question:\n\n${context}\n\nQuestion: ${query}`;
      } else {
        console.warn('No relevant documents found for query:', query);
      }

      // Pass the augmented query and context back to the parent component
      onAugmentedQuery(augmentedQuery, context);
    } catch (error) {
      console.error('Error querying ChromaDB:', error);
      
      // Provide specific error messages
      let errorMessage = 'Error querying the knowledge base. ';
      if (error.message.includes('ChromaDB')) {
        errorMessage += 'Please ensure ChromaDB is running.';
      } else if (error.message.includes('Collection')) {
        errorMessage += 'Please run the RAG setup notebook first.';
      } else if (error.message.includes('timeout')) {
        errorMessage += 'The query took too long to process.';
      } else {
        errorMessage += 'Please check your connection and try again.';
      }
      
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
