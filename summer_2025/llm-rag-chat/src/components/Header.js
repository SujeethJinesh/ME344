import React from 'react';
import './Header.css';

const Header = ({ isMcp }) => (
  <div className='header'>
    <h1>{isMcp ? 'Deep Research Mode' : 'LLM + RAG Slang Translator'}</h1>
  </div>
);

export default Header;
