import { useEffect } from 'react';
import { ChromaClient, OllamaEmbeddingFunction } from 'chromadb';

const Rag = ({ query, onAugmentedQuery }) => {
  const fetchAugmentedQuery = async () => {
    try {
      // Initialize the ChromaDB client
      const client = new ChromaClient({
        path: 'http://localhost:8000',
      });

      const collections = await client.listCollections();
      console.log('Available collections: ', collections);

      const embedder = new OllamaEmbeddingFunction({
        url: 'http://localhost:11434/api/embeddings',
        model: 'nomic-embed-text',
      });

      // Get the collection with the embedding function
      const collection = await client.getCollection({
        name: 'llm_rag_collection',
        embeddingFunction: embedder,
      });

      // Perform the query
      const results = await collection.query({
        queryTexts: [query],
        nResults: 2,
      });

      // Process the results
      let augmentedQuery = query;
      let context = '';

      if (results && results.documents) {
        // Concatenate the retrieved documents to create context
        context = results.documents.flat().join(' ');
        // Augment the user's query with the retrieved context
        augmentedQuery = `Use the following information to answer the question:\n\n${context}\n\nQuestion: ${query}`;
      }

      // Pass the augmented query and context back to the parent component
      onAugmentedQuery(augmentedQuery, context);
    } catch (error) {
      console.error('Error querying ChromaDB:', error);
      onAugmentedQuery(query, ''); // Fallback to original query if error occurs
    }
  };

  useEffect(() => {
    fetchAugmentedQuery();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query]);

  return null; // This component doesn't render anything
};

export default Rag;
