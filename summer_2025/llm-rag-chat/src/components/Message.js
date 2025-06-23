import React from 'react';
import './Message.css';

const Message = ({ text, isUser, isStreaming = false }) => (
  <div className={`message ${isUser ? 'user' : 'bot'} ${isStreaming ? 'streaming' : ''}`}>
    <p>
      {text}
      {isStreaming && (
        <span className="streaming-indicator">
          <span className="cursor">▌</span>
        </span>
      )}
    </p>
  </div>
);

export default Message;
