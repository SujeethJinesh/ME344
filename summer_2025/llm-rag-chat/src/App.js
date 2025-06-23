import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import ChatBox from './components/ChatBox';
import InputBar from './components/InputBar';
import Rag from './components/Rag';
import Sidebar from './components/Sidebar';
import Controls from './components/Controls';
import ErrorBoundary from './components/ErrorBoundary';
import './App.css';

const App = () => {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState('');
  const [context, setContext] = useState('');

  // New state for modes and MCP log
  const [mode, setMode] = useState('rag'); // 'rag' or 'mcp'
  const [logMessages, setLogMessages] = useState([]);

  // Derived state for backward compatibility
  const isRag = mode === 'rag';
  const isMcp = mode === 'mcp';

  useEffect(() => {
    // Only clear when not in progress to avoid inconsistent state
    if (!loading) {
      setContext('');
      setLogMessages([]);
    }
  }, [mode, loading]);

  // --- MCP Deep Research Handler ---
  const handleMcpResearch = async (userInput) => {
    setMessages((prev) => [...prev, { text: userInput, isUser: true }]);
    setLoading(true);
    setLogMessages([]); // Clear previous logs

    const researchUrl = process.env.REACT_APP_RESEARCH_URL || 'http://localhost:8001/research';

    try {
      const response = await fetch(researchUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userInput }),
      });

      if (!response.body) {
        throw new Error('Response body is null.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        const chunk = decoder.decode(value, { stream: true });

        // Process Server-Sent Events
        const events = chunk.split('\n\n').filter(Boolean);
        for (const event of events) {
          if (event.startsWith('data: ')) {
            const dataStr = event.substring(6);
            try {
              const data = JSON.parse(dataStr);

              if (data.type === 'log') {
                setLogMessages((prevLogs) => [...prevLogs, data.content]);
              } else if (data.type === 'answer') {
                setMessages((prev) => [
                  ...prev,
                  { text: data.content, isUser: false },
                ]);
              }
            } catch (parseError) {
              console.error('Failed to parse SSE data:', parseError, 'Raw data:', dataStr);
              // Continue processing other events instead of failing completely
            }
          }
        }
      }
    } catch (error) {
      console.error('Error during deep research:', error);
      const errorMessage =
        'Failed to perform deep research. Please ensure the backend server is running.';
      setMessages((prev) => [...prev, { text: errorMessage, isUser: false }]);
    } finally {
      setLoading(false);
    }
  };

  // --- Original RAG Slang Translator Handler ---
  const handleRagSend = (userInput) => {
    setMessages([...messages, { text: userInput, isUser: true }]);
    setLoading(true);
    setQuery(userInput);
  };

  const handleAugmentedQuery = async (augmentedQuery, contextData) => {
    setContext(contextData); // Display context in the sidebar
    try {
      const ollamaUrl =
        process.env.REACT_APP_OLLAMA_URL || 'http://localhost:11434';
      const modelName = process.env.REACT_APP_OLLAMA_MODEL || 'llama3.1';
      const response = await fetch(`${ollamaUrl}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: modelName,
          prompt: augmentedQuery,
          stream: false,
        }),
      });
      if (!response.ok)
        throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      if (!data.response)
        throw new Error('No response received from the model');
      setMessages((prev) => [...prev, { text: data.response, isUser: false }]);
    } catch (error) {
      console.error('Error fetching LLM response:', error);
      const errorMessage =
        'Error fetching response from the model. Please check your connection and try again.';
      setMessages((prev) => [...prev, { text: errorMessage, isUser: false }]);
    } finally {
      setLoading(false);
      setQuery('');
    }
  };

  // --- Main Send Handler ---
  const handleSend = (userInput) => {
    if (isMcp) {
      handleMcpResearch(userInput);
    } else if (isRag) {
      handleRagSend(userInput);
    } else {
      // Handle case where no mode is selected if you want to support it
      console.warn(
        'No mode selected (RAG or MCP). The message will not be processed.',
      );
    }
  };

  return (
    <div className='app'>
      <ErrorBoundary componentName="Header">
        <Header isMcp={isMcp} />
      </ErrorBoundary>
      
      <ErrorBoundary componentName="Controls">
        <Controls
          isRag={isRag}
          setIsRag={(value) => setMode(value ? 'rag' : 'mcp')}
          isMcp={isMcp}
          setIsMcp={(value) => setMode(value ? 'mcp' : 'rag')}
        />
      </ErrorBoundary>
      
      <div className='main'>
        <ErrorBoundary componentName="ChatBox">
          <ChatBox messages={messages} />
        </ErrorBoundary>
        
        <ErrorBoundary componentName="Sidebar">
          <Sidebar context={context} logMessages={logMessages} isMcp={isMcp} />
        </ErrorBoundary>
      </div>
      
      <ErrorBoundary componentName="InputBar">
        <InputBar onSend={handleSend} loading={loading} />
      </ErrorBoundary>
      
      {isRag && !isMcp && query && (
        <ErrorBoundary componentName="RAG Service">
          <Rag query={query} onAugmentedQuery={handleAugmentedQuery} />
        </ErrorBoundary>
      )}
    </div>
  );
};

export default App;
