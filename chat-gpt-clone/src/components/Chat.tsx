import React, { useState } from 'react';
import axios from 'axios';
import Message, { MessageProps } from './Message';
import ChatInput from './ChatInput';

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<MessageProps[]>([]);
  const [isSending, setIsSending] = useState(false);

  const handleSend = async (message: string) => {
    const newMessage: MessageProps = { text: message, isUser: true };
    setMessages((prevMessages) => [...prevMessages, newMessage]);
    setIsSending(true);

    try {
      const response = await axios.post(
        'http://localhost:11434/api/generate',
        {
          model: 'llama3.1',
          prompt: `Translate the following sentence into gen-z slang. Just output the result with no explanation. This is the sentence: "${message}"`,
          stream: false,
        },
        {
          headers: {
            'Content-Type': 'application/json',
          },
        },
      );

      const botMessage: MessageProps = {
        text: response.data.response,
        isUser: false,
      };
      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: MessageProps = {
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
