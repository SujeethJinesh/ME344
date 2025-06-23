import React from 'react';
import Message from './Message';
import './ChatBox.css';

const ChatBox = ({ messages }) => (
  <div className='chatbox'>
    {messages.map((msg, index) => (
      <Message 
        key={msg.id ? `msg-${msg.id}` : `${msg.text.substring(0, 50)}-${index}-${msg.isUser}`} 
        text={msg.text} 
        isUser={msg.isUser}
        isStreaming={msg.isStreaming || false}
      />
    ))}
  </div>
);

export default ChatBox;
