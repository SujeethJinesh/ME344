import React from 'react';

export interface MessageProps {
  text: string;
  isUser: boolean;
}

const Message: React.FC<MessageProps> = ({ text, isUser }) => {
  return (
    <div className={`Message ${isUser ? 'user' : 'bot'}`}>
      <div className='message-container'>
        <p className='message-label'>{isUser ? 'You' : 'Gen-Z Bot'}</p>
        <p className='message-text'>{text}</p>
      </div>
    </div>
  );
};

export default Message;
