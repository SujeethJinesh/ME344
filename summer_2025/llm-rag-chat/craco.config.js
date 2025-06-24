module.exports = {
  webpack: {
    configure: (webpackConfig) => {
      // Since we installed the optional dependencies, no need to suppress them
      // Just ensure proper module resolution
      webpackConfig.resolve.alias = {
        ...webpackConfig.resolve.alias,
      };

      return webpackConfig;
    },
  },
};