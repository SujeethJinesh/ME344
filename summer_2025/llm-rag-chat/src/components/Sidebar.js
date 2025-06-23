import React from 'react';
import './Sidebar.css';

const Sidebar = ({ context, logMessages, isMcp }) => (
  <div className='sidebar'>
    {isMcp ? (
      <>
        <h2>Research Log</h2>
        <ul className='log-list'>
          {logMessages.map((log, index) => (
            <li key={index}>{log}</li>
          ))}
        </ul>
      </>
    ) : (
      <>
        <h2>Context from Knowledge Base</h2>
        <p>{context}</p>
      </>
    )}
  </div>
);

export default Sidebar;
