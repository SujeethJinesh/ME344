import React, { useState, useCallback, useEffect } from 'react';
import { validateQuery, rateLimiter, CONFIG } from '../utils/validation';
import './InputBar.css';

/**
 * InputBar Component
 * Enhanced input component with validation, rate limiting, and user guidance
 */
const InputBar = ({ onSend, loading }) => {
  const [input, setInput] = useState('');
  const [validationError, setValidationError] = useState(null);
  const [rateLimitWarning, setRateLimitWarning] = useState(null);
  const [charCount, setCharCount] = useState(0);

  // Update character count when input changes
  useEffect(() => {
    setCharCount(input.length);
    
    // Clear validation error when user starts typing
    if (validationError && input.trim().length > 0) {
      setValidationError(null);
    }
  }, [input, validationError]);

  // Clear rate limit warning after timeout
  useEffect(() => {
    if (rateLimitWarning) {
      const timer = setTimeout(() => {
        setRateLimitWarning(null);
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [rateLimitWarning]);

  const handleSend = useCallback((e) => {
    e.preventDefault();
    
    // Clear previous errors
    setValidationError(null);
    setRateLimitWarning(null);

    // Validate input
    const validation = validateQuery(input);
    if (!validation.isValid) {
      setValidationError({
        message: validation.error,
        suggestion: validation.suggestion
      });
      return;
    }

    // Check rate limiting
    if (!rateLimiter.canMakeRequest()) {
      const waitTime = Math.ceil(rateLimiter.getTimeUntilNextRequest() / 1000);
      setRateLimitWarning({
        message: `Please wait ${waitTime} seconds before sending another message`,
        suggestion: 'This helps prevent overwhelming the system'
      });
      return;
    }

    // Don't send if already loading
    if (loading) {
      setValidationError({
        message: 'Please wait for the current request to complete',
        suggestion: 'You can only send one message at a time'
      });
      return;
    }

    // All checks passed - send the message
    try {
      rateLimiter.recordRequest();
      onSend(validation.query);
      setInput('');
      setCharCount(0);
    } catch (error) {
      setValidationError({
        message: 'Failed to send message',
        suggestion: 'Please try again in a moment'
      });
    }
  }, [input, loading, onSend]);

  const handleInputChange = useCallback((e) => {
    const newValue = e.target.value;
    
    // Enforce maximum length at input level
    if (newValue.length <= CONFIG.MAX_QUERY_LENGTH) {
      setInput(newValue);
    }
  }, []);

  const handleKeyPress = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend(e);
    }
  }, [handleSend]);

  // Determine input state styling
  const getInputClassName = () => {
    let className = 'inputbar-input';
    
    if (validationError) {
      className += ' error';
    } else if (charCount > CONFIG.MAX_QUERY_LENGTH * 0.9) {
      className += ' warning';
    } else if (charCount > 0) {
      className += ' active';
    }
    
    return className;
  };

  // Determine character count styling
  const getCharCountClassName = () => {
    let className = 'char-count';
    
    if (charCount > CONFIG.MAX_QUERY_LENGTH * 0.9) {
      className += ' warning';
    }
    if (charCount === CONFIG.MAX_QUERY_LENGTH) {
      className += ' error';
    }
    
    return className;
  };

  return (
    <div className="inputbar-container">
      {/* Validation Error Display */}
      {validationError && (
        <div className="input-error">
          <div className="error-icon">⚠️</div>
          <div className="error-content">
            <div className="error-message">{validationError.message}</div>
            {validationError.suggestion && (
              <div className="error-suggestion">{validationError.suggestion}</div>
            )}
          </div>
        </div>
      )}

      {/* Rate Limit Warning Display */}
      {rateLimitWarning && (
        <div className="input-warning">
          <div className="warning-icon">⏱️</div>
          <div className="warning-content">
            <div className="warning-message">{rateLimitWarning.message}</div>
            {rateLimitWarning.suggestion && (
              <div className="warning-suggestion">{rateLimitWarning.suggestion}</div>
            )}
          </div>
        </div>
      )}

      {/* Main Input Form */}
      <form className="inputbar" onSubmit={handleSend}>
        <div className="input-wrapper">
          <textarea
            className={getInputClassName()}
            value={input}
            onChange={handleInputChange}
            onKeyPress={handleKeyPress}
            disabled={loading}
            placeholder={loading ? 'Processing your request...' : 'Type your question or search query...'}
            rows={1}
            style={{
              minHeight: '44px',
              maxHeight: '120px',
              resize: 'none',
              overflow: 'auto'
            }}
            onInput={(e) => {
              // Auto-resize textarea
              e.target.style.height = 'auto';
              e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
            }}
          />
          
          {/* Character Counter */}
          <div className={getCharCountClassName()}>
            {charCount}/{CONFIG.MAX_QUERY_LENGTH}
          </div>
        </div>

        <button 
          type="submit" 
          className="send-button"
          disabled={loading || charCount === 0 || !!validationError}
          title={loading ? 'Processing...' : charCount === 0 ? 'Enter a message' : 'Send message'}
        >
          {loading ? (
            <div className="loader">
              <div className="spinner"></div>
              <span>Sending...</span>
            </div>
          ) : (
            <>
              <span className="send-icon">📤</span>
              <span>Send</span>
            </>
          )}
        </button>
      </form>

      {/* Helper Text */}
      {!loading && charCount === 0 && !validationError && (
        <div className="input-helper">
          <div className="helper-content">
            💡 <strong>Tips:</strong> Ask specific questions for better results. 
            Try "What is..." or "How does..." to get started.
          </div>
        </div>
      )}

      {/* Service Status Indicator */}
      {loading && (
        <div className="service-status">
          <div className="status-indicator">
            <div className="status-dot processing"></div>
            <span>Processing your request...</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default InputBar;