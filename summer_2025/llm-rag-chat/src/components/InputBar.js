import React, { useState } from 'react';
import './InputBar.css';

const InputBar = ({ onSend, loading }) => {
  const [input, setInput] = useState('');

  const handleSend = (e) => {
    e.preventDefault();
    if (input.trim()) {
      onSend(input);
      setInput('');
    }
  };

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
