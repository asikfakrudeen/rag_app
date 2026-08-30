import React, { useState } from 'react';
import { Send, Upload, Settings2, FileText, Activity } from 'lucide-react';
import './index.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [query, setQuery] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [evidence, setEvidence] = useState([]);
  const [indexStatus, setIndexStatus] = useState('');

  const handleAsk = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    
    const userMsg = { role: 'user', content: query };
    setChatHistory([...chatHistory, userMsg]);
    setLoading(true);
    
    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query, top_k: 5, use_hybrid: true, use_rerank: true })
      });
      
      const data = await res.json();
      
      if (res.ok) {
        setChatHistory(prev => [...prev, { role: 'bot', content: data.answer }]);
        setEvidence(data.evidence);
      } else {
        setChatHistory(prev => [...prev, { role: 'bot', content: 'Error: ' + data.detail }]);
      }
    } catch (err) {
      setChatHistory(prev => [...prev, { role: 'bot', content: 'Connection failed!' }]);
    }
    setLoading(false);
    setQuery('');
  };

  const handleUpload = async (e) => {
    const files = e.target.files;
    if (!files.length) return;
    
    setIndexStatus('Building index... 🔨');
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }
    formData.append('chunk_size', 1500);
    formData.append('overlap', 150);

    try {
      const res = await fetch(`${API_URL}/build-index`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (res.ok) {
        setIndexStatus(`✅ Index built (${data.indexed_chunks} chunks)!`);
      } else {
        setIndexStatus(`❌ Error: ${data.detail}`);
      }
    } catch (err) {
      setIndexStatus('❌ Connection error');
    }
  };

  return (
    <div className="layout">
      {/* Sidebar */}
      <aside className="sidebar glass-panel">
        <div className="brand">
          <Activity className="icon-glow" size={32} />
          <h2>Legal RAG</h2>
        </div>
        
        <div className="sidebar-section">
          <h3><Settings2 size={16} /> Configuration</h3>
          <div className="upload-container">
             <label htmlFor="file-upload" className="upload-btn">
                <Upload size={18} /> Upload Contracts
             </label>
             <input id="file-upload" type="file" multiple accept=".pdf" onChange={handleUpload} style={{display:'none'}}/>
          </div>
          <p className="status-text">{indexStatus}</p>
        </div>
        
        <div className="sidebar-section">
            <h3 style={{color: 'var(--accent-light)'}}>Status</h3>
            <div className="chip">BM25 Hybrid: ON</div>
            <div className="chip">Reranking: ON</div>
        </div>
      </aside>

      {/* Main Chat */}
      <main className="chat-container">
        <div className="messages-area">
          {chatHistory.length === 0 && (
             <div className="empty-state">
                <FileText size={48} className="icon-subtle" />
                <h2>Ask a Legal Question.</h2>
                <p>Upload a contract and start investigating.</p>
             </div>
          )}
          {chatHistory.map((msg, i) => (
            <div key={i} className={`message-bubble ${msg.role}`}>
              {msg.content}
            </div>
          ))}
          {loading && (
            <div className="message-bubble bot typing-indicator">
              <span></span><span></span><span></span>
            </div>
          )}
        </div>
        
        <form onSubmit={handleAsk} className="input-area glass-panel">
          <input 
            type="text" 
            placeholder="Ask about clauses, penalties, or termination..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button type="submit" disabled={loading || !query.trim()} className="send-btn">
             <Send size={20} />
          </button>
        </form>
      </main>

      {/* Inspector Panel */}
      <aside className="inspector-panel glass-panel">
         <h3>🔍 Source Inspection</h3>
         <p className="inspector-desc">See the retrieved chunks powering the answer.</p>
         
         <div className="evidence-list">
            {evidence.length === 0 && <p className="status-text">No chunks fetched yet.</p>}
            {evidence.map((chunk, i) => (
                <div key={i} className="evidence-card">
                  <div className="evidence-meta">
                    <span className="badge">Chunk {i+1}</span>
                    <span className="source-name">{chunk.source} (Pg {chunk.page})</span>
                  </div>
                  <p className="evidence-text">"{chunk.text}"</p>
                  <div className="score-bar">
                    <div className="score-fill" style={{width: `${Math.min(100, Math.max(0, 100 - chunk.distance*100))}%`}}></div>
                  </div>
                  <small>Distance: {chunk.distance.toFixed(4)}</small>
                </div>
            ))}
         </div>
      </aside>
    </div>
  )
}
export default App;
