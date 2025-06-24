import React from 'react';
import './Sidebar.css';

const getStepIcon = (type) => {
  switch (type) {
    case 'info': return '🔵';
    case 'success': return '✅';
    case 'error': return '❌';
    case 'warning': return '⚠️';
    case 'config': return '⚙️';
    case 'query': return '❓';
    case 'timing': return '⏱️';
    case 'metrics': return '📊';
    case 'context': return '📖';
    case 'search': return '🔍';
    case 'analysis': return '🧠';
    case 'web': return '🌐';
    case 'tool': return '🔧';
    case 'workflow': return '⚙️';
    case 'help': return '💡';
    case 'database': return '🗄️';
    case 'embedding': return '🧮';
    case 'vector': return '📐';
    case 'similarity': return '🎯';
    case 'chunking': return '✂️';
    case 'retrieval': return '📋';
    case 'llm': return '🤖';
    case 'generation': return '💭';
    case 'streaming': return '🌊';
    case 'planning': return '📝';
    case 'processing': return '⚡';
    case 'synthesis': return '🔗';
    default: return '📄';
  }
};

const getStepClass = (type) => {
  switch (type) {
    case 'success': return 'step-success';
    case 'error': return 'step-error';
    case 'warning': return 'step-warning';
    case 'config': return 'step-config';
    case 'timing': return 'step-timing';
    case 'metrics': return 'step-metrics';
    case 'help': return 'step-help';
    case 'database': return 'step-database';
    case 'embedding': return 'step-embedding';
    case 'vector': return 'step-vector';
    case 'similarity': return 'step-similarity';
    case 'chunking': return 'step-chunking';
    case 'retrieval': return 'step-retrieval';
    case 'llm': return 'step-llm';
    case 'generation': return 'step-generation';
    case 'streaming': return 'step-streaming';
    case 'planning': return 'step-planning';
    case 'processing': return 'step-processing';
    case 'synthesis': return 'step-synthesis';
    case 'web': return 'step-web';
    case 'analysis': return 'step-analysis';
    case 'tool': return 'step-tool';
    case 'workflow': return 'step-workflow';
    default: return 'step-info';
  }
};


const getCurrentPhase = (processSteps, isMcp) => {
  if (processSteps.length === 0) return 'Initializing...';
  
  const lastStep = processSteps[processSteps.length - 1];
  
  if (isMcp) {
    if (lastStep.message.includes('Planning') || lastStep.message.includes('strategy')) return '🧠 Planning Research';
    if (lastStep.message.includes('search') || lastStep.message.includes('web')) return '🌐 Web Search';
    if (lastStep.message.includes('processing') || lastStep.message.includes('chunk')) return '⚡ Processing Data';
    if (lastStep.message.includes('knowledge') || lastStep.message.includes('vector')) return '🗄️ Updating Knowledge';
    if (lastStep.message.includes('retrieval') || lastStep.message.includes('context')) return '📋 Context Retrieval';
    if (lastStep.message.includes('synthesis') || lastStep.message.includes('answer')) return '🔗 Generating Answer';
    if (lastStep.message.includes('complete') || lastStep.message.includes('finished')) return '✅ Research Complete';
  } else {
    if (lastStep.message.includes('configuration') || lastStep.message.includes('config')) return '⚙️ Configuration';
    if (lastStep.message.includes('ChromaDB') || lastStep.message.includes('database')) return '🗄️ Database Connection';
    if (lastStep.message.includes('embedding') || lastStep.message.includes('Embedding')) return '🧮 Embedding Setup';
    if (lastStep.message.includes('search') || lastStep.message.includes('documents')) return '🔍 Document Search';
    if (lastStep.message.includes('context') || lastStep.message.includes('retrieved')) return '📖 Context Retrieval';
    if (lastStep.message.includes('augment') || lastStep.message.includes('query')) return '🔗 Query Augmentation';
    if (lastStep.message.includes('LLM') || lastStep.message.includes('generation')) return '🤖 Response Generation';
    if (lastStep.message.includes('complete') || lastStep.message.includes('finished')) return '✅ RAG Complete';
  }
  
  return '⚡ Processing...';
};

const Sidebar = ({ context, logMessages, processSteps = [], isMcp }) => (
  <div className='sidebar'>
    <h2>{isMcp ? '🔬 Deep Research Agent' : '🎯 RAG System'}</h2>
    
    {/* Current Phase Indicator */}
    <div className="phase-section">
      <h3>🔄 Current Phase</h3>
      <div className="current-phase">
        {getCurrentPhase(processSteps, isMcp)}
      </div>
    </div>


    {isMcp ? (
      <>
        {/* Research Summary for MCP */}
        <div className="research-section">
          <h3>📋 Research Summary</h3>
          <div className="research-stats">
            <div className="stat-item">
              <span className="stat-label">🔍 Searches:</span>
              <span className="stat-value">
                {processSteps.filter(s => s.message.includes('search') || s.message.includes('web')).length}
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">📄 Documents:</span>
              <span className="stat-value">
                {processSteps.filter(s => s.message.includes('chunk') || s.message.includes('document')).length}
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">⏱️ Time:</span>
              <span className="stat-value">
                {processSteps.length > 0 ? 
                  Math.round((Date.now() - new Date(processSteps[0].timestamp).getTime()) / 1000) + 's' : 
                  '0s'
                }
              </span>
            </div>
          </div>
        </div>

        {/* Research Log - Enhanced */}
        {logMessages.length > 0 && (
          <div className="research-section">
            <h3>📝 Agent Log</h3>
            <ul className='log-list'>
              {logMessages.slice(-5).map((log, index) => (
                <li key={index} className="research-log-item">{log}</li>
              ))}
              {logMessages.length > 5 && (
                <li className="log-more">... and {logMessages.length - 5} more entries</li>
              )}
            </ul>
          </div>
        )}
      </>
    ) : (
      <>
        {/* Context Display - Enhanced */}
        <div className="context-section">
          <h3>📚 Retrieved Context</h3>
          <div className="context-content">
            {context ? (
              <>
                <div className="context-meta">
                  Length: {context.length} characters | 
                  Words: {context.split(' ').length} |
                  Preview:
                </div>
                <p className="context-text">{context.substring(0, 200)}...</p>
                <details className="context-details">
                  <summary>View Full Context</summary>
                  <p className="context-text-full">{context}</p>
                </details>
              </>
            ) : (
              <p className="context-placeholder">No context retrieved yet...</p>
            )}
          </div>
        </div>

        {/* RAG Statistics */}
        <div className="rag-stats-section">
          <h3>📊 RAG Statistics</h3>
          <div className="research-stats">
            <div className="stat-item">
              <span className="stat-label">🔍 Searches:</span>
              <span className="stat-value">
                {processSteps.filter(s => s.message.includes('search') || s.message.includes('query')).length}
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">📖 Context:</span>
              <span className="stat-value">
                {context ? `${context.length} chars` : '0 chars'}
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">⏱️ Time:</span>
              <span className="stat-value">
                {processSteps.length > 0 ? 
                  Math.round((Date.now() - new Date(processSteps[0].timestamp).getTime()) / 1000) + 's' : 
                  '0s'
                }
              </span>
            </div>
          </div>
        </div>
      </>
    )}
    
    {/* Detailed Process Steps - Enhanced */}
    {processSteps.length > 0 && (
      <div className="process-section">
        <h3>⚙️ Detailed Process Log</h3>
        <div className="process-controls">
          <span className="process-count">{processSteps.length} steps logged</span>
        </div>
        <div className='process-steps'>
          {processSteps.slice(-10).map((step, index) => (
            <div key={index} className={`process-step ${getStepClass(step.type)}`}>
              <div className="step-header">
                <span className="step-icon">{getStepIcon(step.type)}</span>
                <span className="step-time">{step.timestamp}</span>
              </div>
              <div className="step-message">{step.message}</div>
              {step.type === 'metrics' && (
                <div className="step-details">
                  <small>📊 Performance metric</small>
                </div>
              )}
              {step.type === 'error' && (
                <div className="step-details error-details">
                  <small>❌ Error occurred - check system status</small>
                </div>
              )}
            </div>
          ))}
          {processSteps.length > 10 && (
            <div className="process-step step-info">
              <div className="step-message">
                ... showing last 10 of {processSteps.length} total steps
              </div>
            </div>
          )}
        </div>
      </div>
    )}
  </div>
);

export default Sidebar;
