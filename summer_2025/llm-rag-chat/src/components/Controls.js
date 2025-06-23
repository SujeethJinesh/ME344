import React from 'react';
import './Controls.css';

const Controls = ({ isRag, setIsRag, isMcp, setIsMcp }) => {
  const handleMcpChange = (e) => {
    const isChecked = e.target.checked;
    setIsMcp(isChecked);
    // When MCP is turned on, RAG must also be turned on.
    if (isChecked) {
      setIsRag(true);
    }
  };

  const handleRagChange = (e) => {
    const isChecked = e.target.checked;
    // You cannot turn off RAG if MCP is on.
    if (!isChecked && isMcp) {
      return;
    }
    setIsRag(isChecked);
  };

  return (
    <div className='controls'>
      <div className='control-item'>
        <label htmlFor='rag-toggle'>RAG (Slang Translator)</label>
        <input
          id='rag-toggle'
          type='checkbox'
          checked={isRag}
          onChange={handleRagChange}
          disabled={isMcp} // Disable changing RAG directly when MCP is on
        />
      </div>
      <div className='control-item'>
        <label htmlFor='mcp-toggle'>MCP (Deep Research)</label>
        <input
          id='mcp-toggle'
          type='checkbox'
          checked={isMcp}
          onChange={handleMcpChange}
        />
      </div>
    </div>
  );
};

export default Controls;
