const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function (app) {
  app.use(
    '/v1', // Proxy any requests with the path /v1
    createProxyMiddleware({
      target: 'http://localhost:5001', // Flask server running on port 5001
      changeOrigin: true,
      secure: false, // If there are any SSL issues, this option might help
    }),
  );
};
