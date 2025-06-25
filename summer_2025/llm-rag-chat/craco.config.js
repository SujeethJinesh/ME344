module.exports = {
  webpack: {
    configure: (webpackConfig) => {
      // Handle ChromaDB optional dependencies that we don't use
      webpackConfig.resolve.fallback = {
        ...webpackConfig.resolve.fallback,
        "cohere-ai": false,
        "chromadb-default-embed": false,
        "openai": false,
        "openai/version": false,
        "@xenova/transformers": false,
        "@google/generative-ai": false,
        "ollama": false
      };
      
      // Suppress warnings for optional ChromaDB dependencies
      webpackConfig.ignoreWarnings = [
        {
          module: /chromadb/,
          message: /Can't resolve/,
        },
      ];

      return webpackConfig;
    },
  },
};