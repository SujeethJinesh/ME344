import React from 'react';
import './StatusIndicator.css';

const StatusIndicator = ({ loading, mode, currentStep }) => {
  const getStatusIcon = () => {
    if (!loading) return '⭕';
    
    if (mode === 'rag') {
      if (currentStep?.includes('ChromaDB')) return '🔌';
      if (currentStep?.includes('embedding')) return '🧮';
      if (currentStep?.includes('search')) return '🔍';
      if (currentStep?.includes('LLM') || currentStep?.includes('Ollama')) return '🤖';
      return '⚡';
    }
    
    if (mode === 'mcp') {
      if (currentStep?.includes('search') || currentStep?.includes('web')) return '🌐';
      if (currentStep?.includes('analysis')) return '🧠';
      if (currentStep?.includes('tool') || currentStep?.includes('MCP')) return '🔧';
      if (currentStep?.includes('workflow')) return '⚙️';
      return '🔬';
    }
    
    return '🔄';
  };

  const getStatusText = () => {
    if (!loading) return 'Ready for queries';
    
    if (mode === 'rag') {
      if (currentStep?.includes('CONFIGURATION')) return 'Setting up RAG pipeline';
      if (currentStep?.includes('DATABASE')) return 'Connecting to ChromaDB';
      if (currentStep?.includes('EMBEDDING')) return 'Configuring embeddings';
      if (currentStep?.includes('SEARCH')) return 'Performing vector search';
      if (currentStep?.includes('RETRIEVAL')) return 'Retrieving context';
      if (currentStep?.includes('AUGMENTATION')) return 'Augmenting query';
      if (currentStep?.includes('GENERATION')) return 'Generating with LLM';
      if (currentStep?.includes('STREAMING')) return 'Streaming response';
      if (currentStep?.includes('Token Generation')) return 'Generating tokens';
      return 'Processing RAG pipeline';
    }
    
    if (mode === 'mcp') {
      if (currentStep?.includes('PLANNING')) return 'Planning research strategy';
      if (currentStep?.includes('WEB SEARCH')) return 'Searching the web';
      if (currentStep?.includes('PROCESSING')) return 'Processing web data';
      if (currentStep?.includes('KNOWLEDGE UPDATE')) return 'Updating knowledge base';
      if (currentStep?.includes('RETRIEVAL')) return 'Retrieving context';
      if (currentStep?.includes('SYNTHESIS')) return 'Synthesizing answer';
      if (currentStep?.includes('MCP TOOL')) return 'Using research tools';
      return 'Deep research in progress';
    }
    
    return 'Processing...';
  };

  const getStatusClass = () => {
    if (!loading) return 'status-ready';
    return mode === 'rag' ? 'status-rag' : 'status-mcp';
  };

  return (
    <div className={`status-indicator ${getStatusClass()}`}>
      <div className="status-icon-container">
        <span className={`status-icon ${loading ? 'spinning' : ''}`}>
          {getStatusIcon()}
        </span>
      </div>
      <div className="status-text">
        <div className="status-mode">
          {mode === 'rag' ? '🎯 RAG Mode' : '🔬 Research Mode'}
        </div>
        <div className="status-message">{getStatusText()}</div>
        {currentStep && loading && (
          <div className="status-detail">
            {currentStep.length > 50 ? 
              `${currentStep.substring(0, 50)}...` : 
              currentStep
            }
          </div>
        )}
      </div>
    </div>
  );
};

export default StatusIndicator;