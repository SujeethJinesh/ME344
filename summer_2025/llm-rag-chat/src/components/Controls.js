import React from 'react';
import './Controls.css';

const Controls = ({ isRag, setIsRag, isMcp, setIsMcp, isStreaming, setIsStreaming }) => {
  // Ensure at least one mode is always selected on component mount
  React.useEffect(() => {
    if (!isRag && !isMcp) {
      console.log('🔧 Controls: No mode selected, defaulting to RAG');
      setIsRag(true);
    }
  }, [isRag, isMcp, setIsRag]);
  const handleMcpChange = (e) => {
    const isChecked = e.target.checked;
    console.log('🔬 MCP toggle clicked:', isChecked);
    if (isChecked) {
      // Switching to MCP mode
      setIsMcp(true);
      setIsRag(false);
    } else if (!isRag) {
      // Don't allow both to be off - default to RAG
      setIsRag(true);
      setIsMcp(false);
    } else {
      setIsMcp(false);
    }
  };

  const handleRagChange = (e) => {
    const isChecked = e.target.checked;
    console.log('🎯 RAG toggle clicked:', isChecked);
    if (isChecked) {
      // Switching to RAG mode
      setIsRag(true);
      setIsMcp(false);
    } else if (!isMcp) {
      // Don't allow both to be off - default to RAG
      setIsRag(true);
      setIsMcp(false);
    } else {
      setIsRag(false);
    }
  };

  return (
    <div className='controls'>
      <div className='control-item'>
        <label htmlFor='rag-toggle'>
          <span className="toggle-icon">🎯</span>
          RAG (Slang Translator)
          {isRag && <span className="status-active">Active</span>}
        </label>
        <input
          id='rag-toggle'
          type='checkbox'
          checked={isRag}
          onChange={handleRagChange}
        />
      </div>
      <div className='control-item'>
        <label htmlFor='mcp-toggle'>
          <span className="toggle-icon">🔬</span>
          MCP (Deep Research)
          {isMcp && <span className="status-active">Active</span>}
        </label>
        <input
          id='mcp-toggle'
          type='checkbox'
          checked={isMcp}
          onChange={handleMcpChange}
        />
      </div>
      <div className='control-item'>
        <label htmlFor='streaming-toggle'>
          <span className="toggle-icon">🌊</span>
          Streaming Mode
          {isStreaming && <span className="status-active">Active</span>}
        </label>
        <input
          id='streaming-toggle'
          type='checkbox'
          checked={isStreaming}
          onChange={(e) => setIsStreaming(e.target.checked)}
        />
      </div>
    </div>
  );
};

export default Controls;
