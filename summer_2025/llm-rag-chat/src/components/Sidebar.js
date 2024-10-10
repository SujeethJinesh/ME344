import React from 'react';
import './Sidebar.css';

const Sidebar = ({ context }) => (
  <div className='sidebar'>
    <h2>Context</h2>
    <p>{context}</p>
  </div>
);

export default Sidebar;
