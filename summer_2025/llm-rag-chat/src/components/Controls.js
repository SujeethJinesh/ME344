import React from 'react';
import './Controls.css';

const Controls = ({ isRag, setIsRag, isMcp, setIsMcp }) => {
  const handleModeToggle = (e) => {
    const isChecked = e.target.checked;
    console.log('🔄 Mode toggle clicked:', isChecked ? 'MCP' : 'RAG');
    if (isChecked) {
      // Switch to MCP mode
      setIsMcp(true);
      setIsRag(false);
    } else {
      // Switch to RAG mode
      setIsRag(true);
      setIsMcp(false);
    }
  };

  return (
    <div className='controls'>
      <div className='control-item mode-toggle'>
        <label htmlFor='mode-toggle'>
          <span className="mode-label">
            <span className="toggle-icon">🎯</span>
            RAG (Slang Translator)
          </span>
          <div className="toggle-switch">
            <input
              id='mode-toggle'
              type='checkbox'
              checked={isMcp}
              onChange={handleModeToggle}
            />
            <span className="slider"></span>
          </div>
          <span className="mode-label">
            <span className="toggle-icon">🔬</span>
            MCP (Deep Research)
          </span>
        </label>
      </div>
    </div>
  );
};

export default Controls;
