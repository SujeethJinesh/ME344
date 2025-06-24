import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import ChatBox from './components/ChatBox';
import InputBar from './components/InputBar';
import Rag from './components/Rag';
import Sidebar from './components/Sidebar';
import Controls from './components/Controls';
import StatusIndicator from './components/StatusIndicator';
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
  const [processSteps, setProcessSteps] = useState([]);

  // Derived state for backward compatibility
  const isRag = mode === 'rag';
  const isMcp = mode === 'mcp';

  useEffect(() => {
    // Only clear when not in progress to avoid inconsistent state
    if (!loading) {
      setContext('');
      setLogMessages([]);
      setProcessSteps([]);
    }
  }, [mode, loading]);

  // Handler for RAG process steps
  const handleProcessStep = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setProcessSteps(prev => [...prev, { message, type, timestamp }]);
  };

  // Get current processing step for status indicator
  const getCurrentStep = () => {
    if (processSteps.length === 0) return null;
    return processSteps[processSteps.length - 1]?.message;
  };

  // --- MCP Deep Research Handler ---
  const handleMcpResearch = async (userInput) => {
    setMessages((prev) => [...prev, { text: userInput, isUser: true }]);
    setLoading(true);
    setLogMessages([]); // Clear previous logs
    setProcessSteps([]); // Clear previous process steps

    const researchUrl = process.env.REACT_APP_RESEARCH_URL || 'http://localhost:8001/research';
    const startTime = Date.now();

    // Initialize detailed logging with enhanced educational visibility
    handleProcessStep("🚀 Initializing Deep Research Agent workflow...", "info");
    handleProcessStep("🔧 Deep Research Agent uses LangGraph for multi-step research", "info");
    handleProcessStep(`📍 Research URL: ${researchUrl}`, "config");
    handleProcessStep(`📝 Research Query: "${userInput}"`, "query");
    handleProcessStep("🧠 Agent will: Plan → Search → Process → Update Knowledge → Retrieve → Synthesize", "planning");

    try {
      handleProcessStep("🔌 Connecting to Deep Research Agent...", "info");
      
      const response = await fetch(researchUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userInput }),
      });

      if (!response.ok) {
        handleProcessStep(`❌ HTTP Error: ${response.status} ${response.statusText}`, "error");
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      if (!response.body) {
        handleProcessStep("❌ Response body is null", "error");
        throw new Error('Response body is null.');
      }

      handleProcessStep("✅ Connection established - starting stream processing", "success");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let chunkCount = 0;
      let totalDataReceived = 0;

      handleProcessStep("📡 Reading response stream...", "info");

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        
        if (value) {
          chunkCount++;
          totalDataReceived += value.length;
          
          const chunk = decoder.decode(value, { stream: true });
          handleProcessStep(`📦 Processing chunk ${chunkCount} (${value.length} bytes)`, "timing");

          // Process Server-Sent Events
          const events = chunk.split('\n\n').filter(Boolean);
          for (const event of events) {
            if (event.startsWith('data: ')) {
              const dataStr = event.substring(6);
              try {
                const data = JSON.parse(dataStr);

                if (data.type === 'log') {
                  const logMessage = data.content;
                  
                  // Enhanced logging with detailed educational context
                  if (logMessage.includes('Planning') || logMessage.includes('strategy')) {
                    handleProcessStep(`🧠 PLANNING PHASE: ${logMessage}`, "planning");
                    handleProcessStep("📚 Educational Note: Planning phase generates search strategy", "info");
                  } else if (logMessage.includes('search') || logMessage.includes('searching') || logMessage.includes('web')) {
                    handleProcessStep(`🌐 WEB SEARCH PHASE: ${logMessage}`, "web");
                    handleProcessStep("📚 Educational Note: Using Tavily API via MCP protocol for web search", "info");
                  } else if (logMessage.includes('processing') || logMessage.includes('chunk')) {
                    handleProcessStep(`⚡ PROCESSING PHASE: ${logMessage}`, "processing");
                    handleProcessStep("📚 Educational Note: Chunking web content for vector storage", "info");
                  } else if (logMessage.includes('knowledge') || logMessage.includes('vector') || logMessage.includes('updating')) {
                    handleProcessStep(`🗄️ KNOWLEDGE UPDATE: ${logMessage}`, "database");
                    handleProcessStep("📚 Educational Note: Adding new knowledge to ChromaDB vector store", "info");
                  } else if (logMessage.includes('retrieval') || logMessage.includes('context')) {
                    handleProcessStep(`📋 RETRIEVAL PHASE: ${logMessage}`, "retrieval");
                    handleProcessStep("📚 Educational Note: Using RAG to retrieve relevant context", "info");
                  } else if (logMessage.includes('synthesis') || logMessage.includes('answer')) {
                    handleProcessStep(`🔗 SYNTHESIS PHASE: ${logMessage}`, "synthesis");
                    handleProcessStep("📚 Educational Note: LLM synthesizing final answer from context", "info");
                  } else if (logMessage.includes('tool') || logMessage.includes('MCP')) {
                    handleProcessStep(`🔧 MCP TOOL: ${logMessage}`, "tool");
                  } else if (logMessage.includes('workflow') || logMessage.includes('step')) {
                    handleProcessStep(`⚙️ WORKFLOW: ${logMessage}`, "workflow");
                  } else if (logMessage.includes('error') || logMessage.includes('failed')) {
                    handleProcessStep(`❌ ERROR: ${logMessage}`, "error");
                  } else if (logMessage.includes('complete') || logMessage.includes('finished')) {
                    handleProcessStep(`✅ SUCCESS: ${logMessage}`, "success");
                  } else {
                    handleProcessStep(`📄 AGENT LOG: ${logMessage}`, "info");
                  }
                  
                  setLogMessages((prevLogs) => [...prevLogs, data.content]);
                } else if (data.type === 'answer') {
                  handleProcessStep("🎯 Generating final research summary...", "info");
                  setMessages((prev) => [
                    ...prev,
                    { text: data.content, isUser: false },
                  ]);
                  handleProcessStep("🎉 Research analysis complete!", "success");
                }
              } catch (parseError) {
                handleProcessStep(`⚠️ Failed to parse stream data: ${dataStr.substring(0, 50)}...`, "warning");
                console.error('Failed to parse SSE data:', parseError, 'Raw data:', dataStr);
                // Continue processing other events instead of failing completely
              }
            }
          }
        }
      }

      const processingTime = Date.now() - startTime;
      handleProcessStep(`📊 Deep Research complete: ${chunkCount} chunks, ${totalDataReceived} bytes in ${processingTime}ms`, "metrics");
    } catch (error) {
      console.error('Error during deep research:', error);
      
      handleProcessStep("❌ Deep Research process failed", "error");
      
      let errorMessage = 'Failed to perform deep research. ';
      if (error.message.includes('HTTP error')) {
        errorMessage += 'Backend server returned an error.';
        handleProcessStep("💡 Solution: Check if Deep Research Agent is running on port 8001", "help");
      } else if (error.message.includes('fetch')) {
        errorMessage += 'Cannot connect to backend server.';
        handleProcessStep("💡 Solution: Start Deep Research Agent with './run_part2.sh'", "help");
      } else {
        errorMessage += error.message;
        handleProcessStep(`💡 Error details: ${error.message}`, "help");
      }
      
      handleProcessStep(`❌ ${errorMessage}`, "error");
      setMessages((prev) => [...prev, { text: errorMessage, isUser: false }]);
    } finally {
      setLoading(false);
    }
  };

  // --- Original RAG Slang Translator Handler ---
  const handleRagSend = (userInput) => {
    setMessages([...messages, { text: userInput, isUser: true }]);
    setLoading(true);
    setProcessSteps([]); // Clear previous process steps
    
    // Enhanced educational logging for RAG process
    handleProcessStep("🎯 Starting RAG (Retrieval Augmented Generation) process...", "info");
    handleProcessStep("📚 Educational Note: RAG combines retrieval from vector database with LLM generation", "info");
    handleProcessStep(`📝 User Query: "${userInput}"`, "query");
    handleProcessStep("🔄 RAG Pipeline: Connect → Embed → Search → Retrieve → Augment → Generate", "planning");
    
    setQuery(userInput);
  };

  const handleAugmentedQuery = async (augmentedQuery, contextData) => {
    setContext(contextData); // Display context in the sidebar
    const ollamaStartTime = Date.now();
    
    // Add placeholder message for streaming response
    const messageId = Date.now();
    
    try {
      const ollamaUrl =
        process.env.REACT_APP_OLLAMA_URL || 'http://localhost:11434';
      const modelName = process.env.REACT_APP_OLLAMA_MODEL || 'llama3.1';
      
      // Detailed Ollama process logging with educational context
      handleProcessStep("🤖 GENERATION PHASE: Starting Ollama LLM generation...", "llm");
      handleProcessStep("📚 Educational Note: Using locally hosted Llama 3.1 model via Ollama", "info");
      handleProcessStep(`📍 Ollama URL: ${ollamaUrl}`, "config");
      handleProcessStep(`📍 Model: ${modelName}`, "config");
      handleProcessStep(`📝 Augmented prompt length: ${augmentedQuery.length} characters`, "metrics");
      handleProcessStep("📚 Educational Note: Prompt includes retrieved context + original query", "info");
      setMessages((prev) => [...prev, { text: '', isUser: false, id: messageId, isStreaming: true }]);
      
      handleProcessStep("🔌 Establishing connection to Ollama API...", "info");
      
      const response = await fetch(`${ollamaUrl}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: modelName,
          prompt: augmentedQuery,
          stream: true, // Enable streaming
        }),
      });
      
      if (!response.ok) {
        handleProcessStep(`❌ Ollama API error: ${response.status} ${response.statusText}`, "error");
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      if (!response.body) {
        handleProcessStep("❌ Ollama response body is null", "error");
        throw new Error('Response body is null.');
      }

      handleProcessStep("✅ Ollama connection established - starting generation", "success");
      handleProcessStep("📚 Educational Note: Using streaming generation for real-time response", "info");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let accumulatedResponse = '';
      let tokenCount = 0;
      let chunkCount = 0;

      handleProcessStep("🌊 STREAMING PHASE: Receiving token stream from LLM...", "streaming");
      handleProcessStep("📚 Educational Note: Each chunk contains generated tokens", "info");

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        
        if (value) {
          chunkCount++;
          const chunk = decoder.decode(value, { stream: !done });
          const lines = chunk.split('\n').filter(line => line.trim() !== '');
          
          handleProcessStep(`📦 Processing LLM chunk ${chunkCount} (${lines.length} lines)`, "timing");
          
          for (const line of lines) {
            try {
              const data = JSON.parse(line);
              if (data.response) {
                tokenCount++;
                accumulatedResponse += data.response;
                
                // Update the streaming message in real-time
                setMessages((prev) => 
                  prev.map((msg) => 
                    msg.id === messageId 
                      ? { ...msg, text: accumulatedResponse }
                      : msg
                  )
                );
                
                // Log progress every 10 tokens with educational context
                if (tokenCount % 10 === 0) {
                  handleProcessStep(`🔄 Token Generation: ${tokenCount} tokens (${accumulatedResponse.length} chars)`, "metrics");
                  if (tokenCount === 10) {
                    handleProcessStep("📚 Educational Note: Tokens are words/word-pieces generated sequentially", "info");
                  }
                }
              }
              
              // Check if the response is complete
              if (data.done) {
                const generationTime = Date.now() - ollamaStartTime;
                const tokensPerSecond = tokenCount > 0 ? (tokenCount / (generationTime / 1000)).toFixed(1) : 0;
                
                handleProcessStep(`🎉 LLM generation complete: ${tokenCount} tokens in ${generationTime}ms (${tokensPerSecond} tokens/sec)`, "metrics");
                
                // Mark streaming as complete
                setMessages((prev) => 
                  prev.map((msg) => 
                    msg.id === messageId 
                      ? { ...msg, isStreaming: false }
                      : msg
                  )
                );
                
                handleProcessStep("✅ Response generation finished", "success");
                done = true;
                break;
              }
            } catch (parseError) {
              handleProcessStep(`⚠️ Failed to parse LLM chunk: ${line.substring(0, 50)}...`, "warning");
              console.error('Failed to parse streaming data:', parseError, 'Raw line:', line);
              // Continue processing other lines instead of failing completely
            }
          }
        }
      }
      
      if (!accumulatedResponse) {
        throw new Error('No response received from the model');
      }
    } catch (error) {
      console.error('Error fetching LLM response:', error);
      
      handleProcessStep("❌ LLM generation failed", "error");
      
      let errorMessage = 'Error fetching response from the model. ';
      if (error.message.includes('HTTP error')) {
        errorMessage += 'Ollama server returned an error.';
        handleProcessStep("💡 Solution: Check if Ollama is running with 'ollama serve'", "help");
      } else if (error.message.includes('fetch')) {
        errorMessage += 'Cannot connect to Ollama server.';
        handleProcessStep("💡 Solution: Start Ollama server on port 11434", "help");
      } else if (error.message.includes('No response')) {
        errorMessage += 'Model did not generate any response.';
        handleProcessStep("💡 Solution: Try a different query or check model availability", "help");
      } else {
        errorMessage += 'Please check your connection and try again.';
        handleProcessStep(`💡 Error details: ${error.message}`, "help");
      }
      
      handleProcessStep(`❌ ${errorMessage}`, "error");
      
      // Replace the streaming message with error message
      setMessages((prev) => 
        prev.map((msg) => 
          msg.id === messageId 
            ? { ...msg, text: errorMessage, isStreaming: false }
            : msg
        )
      );
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
      
      <ErrorBoundary componentName="StatusIndicator">
        <StatusIndicator 
          loading={loading}
          mode={mode}
          currentStep={getCurrentStep()}
        />
      </ErrorBoundary>
      
      <div className='main'>
        <ErrorBoundary componentName="ChatBox">
          <ChatBox messages={messages} />
        </ErrorBoundary>
        
        <ErrorBoundary componentName="Sidebar">
          <Sidebar 
            context={context} 
            logMessages={logMessages} 
            processSteps={processSteps}
            isMcp={isMcp} 
          />
        </ErrorBoundary>
      </div>
      
      <ErrorBoundary componentName="InputBar">
        <InputBar onSend={handleSend} loading={loading} />
      </ErrorBoundary>
      
      {isRag && !isMcp && query && (
        <ErrorBoundary componentName="RAG Service">
          <Rag 
            query={query} 
            onAugmentedQuery={handleAugmentedQuery}
            onProcessStep={handleProcessStep}
          />
        </ErrorBoundary>
      )}
    </div>
  );
};

export default App;
