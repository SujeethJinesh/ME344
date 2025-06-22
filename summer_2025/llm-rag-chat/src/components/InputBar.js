import React, { useState, useCallback } from 'react';
import './InputBar.css';

const InputBar = ({ onSend, loading }) => {
  const [input, setInput] = useState('');
  const [lastSubmitTime, setLastSubmitTime] = useState(0);

  // Debounce requests to prevent spam
  const handleSend = useCallback((e) => {
    e.preventDefault();
    
    const now = Date.now();
    const minInterval = 1000; // Minimum 1 second between requests
    
    if (now - lastSubmitTime < minInterval) {
      console.warn('Please wait before sending another message');
      return;
    }
    
    if (input.trim() && !loading) {
      setLastSubmitTime(now);
      onSend(input);
      setInput('');
    }
  }, [input, loading, onSend, lastSubmitTime]);

  return (
    <form className='inputbar' onSubmit={handleSend}>
      <input
        type='text'
        value={input}
        onChange={(e) => setInput(e.target.value)}
        disabled={loading}
        placeholder='Type your message...'
      />
      <button type='submit' disabled={loading}>
        {loading ? <div className='loader'></div> : 'Send'}
      </button>
    </form>
  );
};

export default InputBar;
