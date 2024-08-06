import React, { useState, useEffect, useRef } from 'react';
import { ClipLoader } from 'react-spinners';

interface ChatInputProps {
  onSend: (message: string) => void;
  isSending: boolean;
}

const ChatInput: React.FC<ChatInputProps> = ({ onSend, isSending }) => {
  const [input, setInput] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, [isSending]);

  const handleSend = () => {
    if (input.trim() !== '') {
      onSend(input);
      setInput('');
    }
  };

  return (
    <div className='ChatInput'>
      <input
        ref={inputRef}
        type='text'
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyPress={(e) => e.key === 'Enter' && handleSend()}
        placeholder='Type a message...'
        disabled={isSending}
      />
      <button onClick={handleSend} disabled={isSending}>
        {isSending ? <ClipLoader color='#ffffff' size={20} /> : 'Send'}
      </button>
    </div>
  );
};

export default ChatInput;
