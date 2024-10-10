import React from 'react';
import './Message.css';

const Message = ({ text, isUser }) => (
  <div className={`message ${isUser ? 'user' : 'bot'}`}>
    <p>{text}</p>
  </div>
);

export default Message;
