import React from 'react';
import { Routes, Route, Link } from 'react-router-dom';
import './App.css';
import Home from './components/Home';
import DetectionResults from './components/DetectionResults';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <div className="header-content">
          <h1><Link to="/">Object Detection System</Link></h1>
          <nav>
            <ul>
              <li><Link to="/">Home</Link></li>
              <li><a href="https://github.com/yourusername/object-detection-project" target="_blank" rel="noopener noreferrer">GitHub</a></li>
            </ul>
          </nav>
        </div>
      </header>

      <main className="App-main">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/results/:id" element={<DetectionResults />} />
        </Routes>
      </main>

      <footer className="App-footer">
        <div className="footer-content">
          <p>Custom Object Detection System &copy; {new Date().getFullYear()}</p>
          <p>
            <a href="https://github.com/yourusername/object-detection-project" target="_blank" rel="noopener noreferrer">View on GitHub</a>
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;