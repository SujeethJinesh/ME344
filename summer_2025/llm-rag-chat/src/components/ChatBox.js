import React from 'react';
import Message from './Message';
import './ChatBox.css';

const ChatBox = ({ messages }) => (
  <div className='chatbox'>
    {messages.map((msg, index) => (
      <Message key={index} text={msg.text} isUser={msg.isUser} />
    ))}
  </div>
);

export default ChatBox;
