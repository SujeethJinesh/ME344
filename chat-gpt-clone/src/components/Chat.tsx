import React, { useState } from 'react';
import axios from 'axios';
import Message from './Message';
import ChatInput from './ChatInput';

interface Message {
  text: string;
  isUser: boolean;
}

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isSending, setIsSending] = useState(false);

  const handleSend = async (message: string) => {
    const newMessage: Message = { text: message, isUser: true };
    setMessages((prevMessages) => [...prevMessages, newMessage]);
    setIsSending(true);

    try {
      const response = await axios.post('YOUR_API_ENDPOINT', { message });
      const botMessage: Message = { text: response.data, isUser: false };
      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        text: 'Failed to send message. Please try again.',
        isUser: false,
      };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    } finally {
      setIsSending(false);
    }
  };

  return (
    <div className='Chat'>
      <div className='ChatMessages'>
        {messages.map((msg, index) => (
          <Message key={index} text={msg.text} isUser={msg.isUser} />
        ))}
      </div>
      <ChatInput onSend={handleSend} isSending={isSending} />
    </div>
  );
};

export default Chat;
