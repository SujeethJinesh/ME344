/**
 * ME344 Frontend Validation Utilities
 * Provides consistent validation and error handling across the React application
 */

// ===================================================================
// CONFIGURATION
// ===================================================================

export const CONFIG = {
  MAX_QUERY_LENGTH: 1000,
  MIN_QUERY_LENGTH: 3,
  REQUEST_TIMEOUT: 30000, // 30 seconds
  SSE_TIMEOUT: 60000, // 60 seconds
  REQUIRED_SERVICES: {
    CHROMA: process.env.REACT_APP_CHROMA_URL || 'http://localhost:8000',
    OLLAMA: process.env.REACT_APP_OLLAMA_URL || 'http://localhost:11434',
    RESEARCH: process.env.REACT_APP_RESEARCH_URL || 'http://localhost:8001/research'
  }
};

// ===================================================================
// VALIDATION FUNCTIONS
// ===================================================================

/**
 * Validate user query input
 * @param {string} query - User input query
 * @returns {object} - Validation result with isValid and error message
 */
export const validateQuery = (query) => {
  if (!query || typeof query !== 'string') {
    return {
      isValid: false,
      error: 'Query is required',
      suggestion: 'Please enter a question or search term'
    };
  }

  const trimmedQuery = query.trim();
  
  if (trimmedQuery.length === 0) {
    return {
      isValid: false,
      error: 'Query cannot be empty',
      suggestion: 'Please enter a question or search term'
    };
  }

  if (trimmedQuery.length < CONFIG.MIN_QUERY_LENGTH) {
    return {
      isValid: false,
      error: `Query too short (minimum ${CONFIG.MIN_QUERY_LENGTH} characters)`,
      suggestion: 'Please provide more specific details in your query'
    };
  }

  if (trimmedQuery.length > CONFIG.MAX_QUERY_LENGTH) {
    return {
      isValid: false,
      error: `Query too long (maximum ${CONFIG.MAX_QUERY_LENGTH} characters)`,
      suggestion: 'Please shorten your query or break it into multiple questions'
    };
  }

  // Check for potentially problematic characters
  const suspiciousPatterns = [
    /[<>\"']/g,  // HTML/script injection
    /javascript:/gi, // JavaScript URLs
    /data:/gi // Data URLs
  ];

  for (const pattern of suspiciousPatterns) {
    if (pattern.test(trimmedQuery)) {
      return {
        isValid: false,
        error: 'Query contains invalid characters',
        suggestion: 'Please use only letters, numbers, and basic punctuation'
      };
    }
  }

  return {
    isValid: true,
    query: trimmedQuery
  };
};

/**
 * Validate environment configuration
 * @returns {object} - Validation result with missing services and warnings
 */
export const validateEnvironment = () => {
  const warnings = [];
  const errors = [];

  // Check if required environment variables are set
  Object.entries(CONFIG.REQUIRED_SERVICES).forEach(([service, url]) => {
    if (!url || url === 'undefined') {
      errors.push(`${service} service URL not configured`);
    } else {
      try {
        new URL(url);
      } catch (e) {
        errors.push(`${service} service URL is invalid: ${url}`);
      }
    }
  });

  // Check browser capabilities
  if (!window.fetch) {
    errors.push('Browser does not support fetch API');
  }

  if (!window.EventSource) {
    warnings.push('Browser does not support Server-Sent Events (some features may not work)');
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings
  };
};

// ===================================================================
// ERROR CLASSIFICATION
// ===================================================================

/**
 * Classify error types and provide user-friendly messages
 * @param {Error|string} error - Error object or message
 * @param {string} context - Context where error occurred
 * @returns {object} - Classified error with user message and suggestions
 */
export const classifyError = (error, context = 'general') => {
  const errorMessage = error?.message || error?.toString() || 'Unknown error';
  const lowerMessage = errorMessage.toLowerCase();

  // Network errors
  if (lowerMessage.includes('fetch') || lowerMessage.includes('network') || 
      lowerMessage.includes('connection') || error?.name === 'TypeError') {
    return {
      type: 'network',
      title: 'Connection Error',
      message: 'Unable to connect to the service',
      suggestions: [
        'Check your internet connection',
        'Verify that all services are running',
        'Try refreshing the page'
      ],
      technical: errorMessage
    };
  }

  // Timeout errors
  if (lowerMessage.includes('timeout') || lowerMessage.includes('aborted')) {
    return {
      type: 'timeout',
      title: 'Request Timeout',
      message: 'The request took too long to complete',
      suggestions: [
        'Try a simpler or shorter query',
        'Check if the services are responding',
        'Wait a moment and try again'
      ],
      technical: errorMessage
    };
  }

  // Server errors (5xx)
  if (lowerMessage.includes('500') || lowerMessage.includes('server error') ||
      lowerMessage.includes('internal error')) {
    return {
      type: 'server',
      title: 'Server Error',
      message: 'The server encountered an error processing your request',
      suggestions: [
        'Try again in a moment',
        'Check if all services are running properly',
        'Contact support if the problem persists'
      ],
      technical: errorMessage
    };
  }

  // Client errors (4xx)
  if (lowerMessage.includes('400') || lowerMessage.includes('404') || 
      lowerMessage.includes('unauthorized') || lowerMessage.includes('forbidden')) {
    return {
      type: 'client',
      title: 'Request Error',
      message: 'There was a problem with your request',
      suggestions: [
        'Check your input and try again',
        'Make sure you have the required permissions',
        'Verify the service is available'
      ],
      technical: errorMessage
    };
  }

  // Parse errors
  if (lowerMessage.includes('json') || lowerMessage.includes('parse') ||
      lowerMessage.includes('syntax')) {
    return {
      type: 'parse',
      title: 'Data Format Error',
      message: 'Received invalid data from the server',
      suggestions: [
        'Try your request again',
        'Check if the service is running correctly',
        'Contact support if the problem persists'
      ],
      technical: errorMessage
    };
  }

  // Service-specific errors
  if (context === 'rag' || lowerMessage.includes('chromadb') || lowerMessage.includes('chroma')) {
    return {
      type: 'service',
      title: 'RAG Service Error',
      message: 'The knowledge base service is not responding',
      suggestions: [
        'Make sure ChromaDB is running on port 8000',
        'Check if the database has been initialized',
        'Try running the setup notebook first'
      ],
      technical: errorMessage
    };
  }

  if (context === 'mcp' || lowerMessage.includes('mcp') || lowerMessage.includes('research')) {
    return {
      type: 'service',
      title: 'Research Agent Error',
      message: 'The research service is not responding',
      suggestions: [
        'Make sure the Deep Research Agent is running on port 8001',
        'Check if the MCP server is running on port 8002',
        'Verify that the TAVILY_API_KEY is set'
      ],
      technical: errorMessage
    };
  }

  // Generic error
  return {
    type: 'unknown',
    title: 'Unexpected Error',
    message: 'An unexpected error occurred',
    suggestions: [
      'Try refreshing the page',
      'Check your internet connection',
      'Contact support if the problem persists'
    ],
    technical: errorMessage
  };
};

// ===================================================================
// SERVICE HEALTH CHECKS
// ===================================================================

/**
 * Check if a service is reachable
 * @param {string} url - Service URL to check
 * @param {number} timeout - Timeout in milliseconds
 * @returns {Promise<object>} - Health check result
 */
export const checkServiceHealth = async (url, timeout = 5000) => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    const response = await fetch(url, {
      method: 'GET',
      signal: controller.signal,
      headers: {
        'Accept': 'application/json'
      }
    });

    clearTimeout(timeoutId);

    return {
      isHealthy: response.ok,
      status: response.status,
      url: url
    };
  } catch (error) {
    return {
      isHealthy: false,
      error: error.message,
      url: url
    };
  }
};

/**
 * Check health of all required services
 * @returns {Promise<object>} - Overall health status
 */
export const checkAllServices = async () => {
  const healthChecks = await Promise.allSettled(
    Object.entries(CONFIG.REQUIRED_SERVICES).map(async ([name, url]) => {
      const health = await checkServiceHealth(url);
      return { name, ...health };
    })
  );

  const results = healthChecks.map(result => 
    result.status === 'fulfilled' ? result.value : { 
      name: 'unknown', 
      isHealthy: false, 
      error: result.reason?.message || 'Health check failed' 
    }
  );

  const healthyServices = results.filter(r => r.isHealthy);
  const unhealthyServices = results.filter(r => !r.isHealthy);

  return {
    isAllHealthy: unhealthyServices.length === 0,
    healthyCount: healthyServices.length,
    totalCount: results.length,
    services: results,
    summary: `${healthyServices.length}/${results.length} services healthy`
  };
};

// ===================================================================
// RETRY LOGIC
// ===================================================================

/**
 * Retry a function with exponential backoff
 * @param {Function} fn - Function to retry
 * @param {number} maxRetries - Maximum number of retries
 * @param {number} baseDelay - Base delay in milliseconds
 * @returns {Promise} - Result of the function or final error
 */
export const retryWithBackoff = async (fn, maxRetries = 3, baseDelay = 1000) => {
  let lastError;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      
      if (attempt === maxRetries) {
        throw error;
      }

      // Calculate delay with exponential backoff
      const delay = baseDelay * Math.pow(2, attempt);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }

  throw lastError;
};

// ===================================================================
// RATE LIMITING
// ===================================================================

class RateLimiter {
  constructor(maxRequests = 5, windowMs = 60000) {
    this.maxRequests = maxRequests;
    this.windowMs = windowMs;
    this.requests = [];
  }

  canMakeRequest() {
    const now = Date.now();
    
    // Remove old requests outside the window
    this.requests = this.requests.filter(time => now - time < this.windowMs);
    
    // Check if we're under the limit
    return this.requests.length < this.maxRequests;
  }

  recordRequest() {
    this.requests.push(Date.now());
  }

  getTimeUntilNextRequest() {
    if (this.canMakeRequest()) {
      return 0;
    }

    const oldestRequest = Math.min(...this.requests);
    return this.windowMs - (Date.now() - oldestRequest);
  }
}

export const rateLimiter = new RateLimiter();

// ===================================================================
// EXPORT ALL UTILITIES
// ===================================================================

export default {
  CONFIG,
  validateQuery,
  validateEnvironment,
  classifyError,
  checkServiceHealth,
  checkAllServices,
  retryWithBackoff,
  rateLimiter
};