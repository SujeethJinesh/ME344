import React from 'react';

interface MessageProps {
  text: string;
  isUser: boolean;
}

const Message: React.FC<MessageProps> = ({ text, isUser }) => {
  return (
    <div className={`Message ${isUser ? 'user' : 'bot'}`}>
      <p>{text}</p>
    </div>
  );
};

export default Message;
