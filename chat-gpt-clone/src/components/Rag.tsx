import React from 'react';
import { ChromaClient, OllamaEmbeddingFunction, IncludeEnum } from 'chromadb';
// import { IncludeEnum } from 'chromadb/dist/chromadb';

// enum IncludeEnum {
//   Documents = 'documents',
//   Embeddings = 'embeddings',
//   Metadatas = 'metadatas',
//   Distances = 'distances',
// }

interface RagProps {
  query: string;
  onAugmentedQuery: (augmentedQuery: string) => void;
}

const Rag: React.FC<RagProps> = ({ query, onAugmentedQuery }) => {
  const fetchAugmentedQuery = async () => {
    try {
      // Initialize the ChromaDB client
      const client = new ChromaClient({
        path: 'http://localhost:3000',
      });

      // Dummy embedding function (no-op)
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
        include: [IncludeEnum.Documents],
      });

      // Process the results
      let augmentedQuery = query;

      if (results && results.documents) {
        // Concatenate the retrieved documents to create context
        const context = results.documents.flat().join(' ');
        // Augment the user's query with the retrieved context
        augmentedQuery = `Use the following information to answer the question:\n\n${context}\n\nQuestion: ${query}`;
      }

      // Pass the augmented query back to the parent component
      onAugmentedQuery(augmentedQuery);
    } catch (error) {
      console.error('Error querying ChromaDB:', error);
      onAugmentedQuery(query); // Fallback to original query if error occurs
    }
  };

  // Fetch the augmented query when the component mounts or query changes
  React.useEffect(() => {
    fetchAugmentedQuery();
  }, [query]);

  return null; // This component doesn't render anything
};

export default Rag;
