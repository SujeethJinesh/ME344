import React, { useState } from 'react';
import Header from './components/Header';
import ChatBox from './components/ChatBox';
import InputBar from './components/InputBar';
import Rag from './components/Rag';
import Sidebar from './components/Sidebar';
import './App.css';

const App = () => {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState('');
  const [context, setContext] = useState('');

  const handleSend = (userInput) => {
    setMessages([...messages, { text: userInput, isUser: true }]);
    setLoading(true);
    setQuery(userInput);
  };

  const handleAugmentedQuery = async (augmentedQuery, contextData) => {
    setContext(contextData); // Display context in the sidebar

    try {
      // Call the LLM API with the augmented query
      const response = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },

        body: JSON.stringify({
          model: 'llama3.1',
          prompt: augmentedQuery,
          stream: false,
        }),
      });

      const data = await response.json();

      // Update messages with LLM response
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: data.response, isUser: false },
      ]);
    } catch (error) {
      console.error('Error fetching LLM response:', error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: 'Error fetching response from the model.', isUser: false },
      ]);
    } finally {
      setLoading(false);
      setQuery('');
    }
  };

  return (
    <div className='app'>
      <Header />
      <div className='main'>
        <ChatBox messages={messages} />
        <Sidebar context={context} />
      </div>
      <InputBar onSend={handleSend} loading={loading} />
      {query && <Rag query={query} onAugmentedQuery={handleAugmentedQuery} />}
    </div>
  );
};

export default App;
