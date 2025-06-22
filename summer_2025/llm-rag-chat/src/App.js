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
      // Use environment variables for configuration
      const ollamaUrl = process.env.REACT_APP_OLLAMA_URL || 'http://localhost:11434';
      const modelName = process.env.REACT_APP_OLLAMA_MODEL || 'llama3.1';

      // Call the LLM API with the augmented query
      const response = await fetch(`${ollamaUrl}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: modelName,
          prompt: augmentedQuery,
          stream: false,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (!data.response) {
        throw new Error('No response received from the model');
      }

      // Update messages with LLM response
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: data.response, isUser: false },
      ]);
    } catch (error) {
      console.error('Error fetching LLM response:', error);
      
      // Provide specific error messages
      let errorMessage = 'Error fetching response from the model. ';
      if (error.message.includes('status: 404')) {
        errorMessage += 'Model not found. Please ensure Ollama is running and the model is available.';
      } else if (error.message.includes('Failed to fetch')) {
        errorMessage += 'Cannot connect to Ollama. Please ensure Ollama is running on the correct port.';
      } else {
        errorMessage += 'Please check your connection and try again.';
      }
      
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: errorMessage, isUser: false },
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
