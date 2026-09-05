import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { Send, Upload, Settings2, FileText, Activity, ChevronLeft, ChevronRight, X, Search, PanelRightClose, PanelRightOpen, FileSpreadsheet, FileType, FileCode, ExternalLink, Trash2 } from 'lucide-react';
import './index.css';

const API_URL = 'http://localhost:8000';

// Maps extension → { label, color } for the file-type badge
const FILE_TYPE_META = {
  '.pdf':  { label: 'PDF',  color: '#ef4444' },
  '.docx': { label: 'DOC', color: '#3b82f6' },
  '.txt':  { label: 'TXT', color: '#94a3b8' },
  '.md':   { label: 'MD',  color: '#a78bfa' },
  '.csv':  { label: 'CSV', color: '#10b981' },
  '.xlsx': { label: 'XLS', color: '#f59e0b' },
};

function getFileMeta(filename) {
  const ext = '.' + filename.split('.').pop().toLowerCase();
  return FILE_TYPE_META[ext] || { label: ext.replace('.','').toUpperCase(), color: '#94a3b8' };
}

function ToggleSwitch({ enabled, onToggle, label }) {
  return (
    <div className="toggle-row" onClick={onToggle}>
      <span className="toggle-label">{label}</span>
      <div className={`toggle-track ${enabled ? 'enabled' : ''}`}>
        <div className="toggle-thumb" />
      </div>
    </div>
  );
}

/* ── Chunk Detail Modal ─────────────────────────────────────── */
function ChunkModal({ chunk, index, onClose }) {
  // Close on Escape key
  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  const relevance = Math.min(100, Math.max(0, 100 - chunk.distance * 100));

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-panel" onClick={(e) => e.stopPropagation()}>

        {/* Header */}
        <div className="modal-header">
          <div className="modal-title-group">
            <span className="badge modal-badge">Chunk {index + 1}</span>
            <span className="modal-source">{chunk.source} · Page {chunk.page}</span>
          </div>
          <button className="icon-btn" onClick={onClose} title="Close (Esc)">
            <X size={18} />
          </button>
        </div>

        {/* Relevance bar */}
        <div className="modal-relevance">
          <span className="modal-relevance-label">Relevance</span>
          <div className="modal-score-track">
            <div className="modal-score-fill" style={{ width: `${relevance}%` }} />
          </div>
          <span className="modal-relevance-pct">{relevance.toFixed(1)}%</span>
        </div>

        {/* Body — full text */}
        <div className="modal-body">
          <p className="modal-text">{chunk.text}</p>
        </div>

        {/* Footer */}
        <div className="modal-footer">
          <span className="modal-distance">Distance score: {chunk.distance.toFixed(6)}</span>
          <button className="icon-btn" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}

/* ── Main App ───────────────────────────────────────────────── */
function App() {
  const [query, setQuery]               = useState('');
  const [chatHistory, setChatHistory]   = useState([]);
  const [loading, setLoading]           = useState(false);
  const [evidence, setEvidence]         = useState([]);
  const [indexStatus, setIndexStatus]   = useState('');
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [sidebarOpen, setSidebarOpen]   = useState(true);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [useHybrid, setUseHybrid]       = useState(true);
  const [useRerank, setUseRerank]       = useState(true);
  const [logoAnimating, setLogoAnimating] = useState(false);
  const [selectedChunk, setSelectedChunk] = useState(null); // { chunk, index }
  const [clearingChat, setClearingChat] = useState(false);

  const fileInputRef  = useRef(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  /* Clear chat */
  const handleClearChat = () => {
    if (!chatHistory.length) return;
    setClearingChat(true);
    setTimeout(() => {
      setChatHistory([]);
      setEvidence([]);
      setClearingChat(false);
    }, 550);
  };

  /* Logo click animation */
  const handleLogoClick = () => {
    if (logoAnimating) return;
    setLogoAnimating(true);
    setTimeout(() => setLogoAnimating(false), 800);
  };

  const handleAsk = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMsg = { role: 'user', content: query };
    setChatHistory(prev => [...prev, userMsg]);
    setLoading(true);
    setQuery('');
    setTimeout(scrollToBottom, 50);

    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query, top_k: 5, use_hybrid: useHybrid, use_rerank: useRerank })
      });
      const data = await res.json();
      if (res.ok) {
        setChatHistory(prev => [...prev, { role: 'bot', content: data.answer }]);
        setEvidence(data.evidence);
        if (!inspectorOpen) setInspectorOpen(true);
      } else {
        setChatHistory(prev => [...prev, { role: 'bot', content: '❌ Error: ' + data.detail }]);
      }
    } catch (err) {
      setChatHistory(prev => [...prev, { role: 'bot', content: '❌ Connection failed. Is the backend running?' }]);
    }
    setLoading(false);
    setTimeout(scrollToBottom, 100);
  };

  const handleUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;
    fileInputRef.current.value = '';

    const newFiles = files.map(f => ({ name: f.name, file: f, status: 'uploading' }));
    setUploadedFiles(prev => [...prev, ...newFiles]);
    setIndexStatus('Building index... 🔨');

    const formData = new FormData();
    files.forEach(f => formData.append('files', f));
    formData.append('chunk_size', 1500);
    formData.append('overlap', 150);

    try {
      const res = await fetch(`${API_URL}/build-index`, { method: 'POST', body: formData });
      const data = await res.json();
      if (res.ok) {
        setIndexStatus(`✅ Index built (${data.indexed_chunks} chunks)!`);
        setUploadedFiles(prev =>
          prev.map(f => files.find(uf => uf.name === f.name) ? { ...f, status: 'indexed' } : f)
        );
      } else {
        setIndexStatus(`❌ Error: ${data.detail}`);
        setUploadedFiles(prev =>
          prev.map(f => files.find(uf => uf.name === f.name) ? { ...f, status: 'error' } : f)
        );
      }
    } catch (err) {
      setIndexStatus('❌ Connection error');
      setUploadedFiles(prev =>
        prev.map(f => files.find(uf => uf.name === f.name) ? { ...f, status: 'error' } : f)
      );
    }
  };

  const removeFile = (fileName) => {
    setUploadedFiles(prev => prev.filter(f => f.name !== fileName));
    if (uploadedFiles.length <= 1) {
      setIndexStatus('');
      setEvidence([]);
    }
  };

  const clearIndex = async () => {
    try {
      await fetch(`${API_URL}/clear-index`, { method: 'DELETE' });
    } catch (_) {}
    setUploadedFiles([]);
    setEvidence([]);
    setIndexStatus('🗑️ Index cleared.');
    setTimeout(() => setIndexStatus(''), 3000);
  };

  return (
    <div className={`layout sidebar-${sidebarOpen ? 'open' : 'closed'} inspector-${inspectorOpen ? 'open' : 'closed'}`}>

      {/* ═══ SIDEBAR ═══ */}
      <aside className={`sidebar glass-panel ${sidebarOpen ? 'sidebar-visible' : 'sidebar-hidden'}`}>

        {/* ── Brand / Logo ── */}
        <div className="brand" onClick={handleLogoClick} title="Legal RAG" style={{ cursor: 'pointer' }}>
          <div className={`logo-wrap ${logoAnimating ? 'logo-burst' : ''}`}>
            <Activity className="icon-glow logo-icon" size={28} />
            {logoAnimating && (
              <>
                <span className="ripple r1" />
                <span className="ripple r2" />
                <span className="ripple r3" />
              </>
            )}
          </div>
          <h2 className={logoAnimating ? 'logo-text-flash' : ''}>Legal RAG</h2>
        </div>

        {/* Configuration Section */}
        <div className="sidebar-section">
          <h3><Settings2 size={14} /> Configuration</h3>
          <ToggleSwitch enabled={useHybrid} onToggle={() => setUseHybrid(v => !v)} label="BM25 Hybrid" />
          <ToggleSwitch enabled={useRerank} onToggle={() => setUseRerank(v => !v)} label="Reranking" />
        </div>

        {/* Upload Section */}
        <div className="sidebar-section">
          <h3><Upload size={14} /> Upload Contracts</h3>
          <label htmlFor="file-upload" className="upload-btn">
            <Upload size={16} /> Choose Files
          </label>
          <input
            id="file-upload"
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.docx,.txt,.md,.csv,.xlsx"
            onChange={handleUpload}
            style={{ display: 'none' }}
          />

          {uploadedFiles.length > 0 && (
            <div className="file-list">
              {uploadedFiles.map((f, i) => {
                const meta = getFileMeta(f.name);
                return (
                  <div key={i} className={`file-item ${f.status}`}>
                    <span
                      className="file-type-badge"
                      style={{ background: meta.color + '22', color: meta.color, borderColor: meta.color + '55' }}
                    >
                      {meta.label}
                    </span>
                    <span className="file-name" title={f.name}>{f.name}</span>
                    <div className="file-status-dot" title={f.status} />
                    <button className="file-remove-btn" onClick={() => removeFile(f.name)} title="Remove">
                      <X size={13} />
                    </button>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Status Section */}
        <div className="sidebar-section">
          <h3 style={{ color: 'var(--accent-light)' }}>Status</h3>
          {indexStatus
            ? <p className="status-text">{indexStatus}</p>
            : <p className="status-text" style={{ color: 'var(--text-secondary)' }}>No index built yet.</p>
          }
          <div className="status-chips">
            <div className={`chip ${useHybrid ? 'chip-active' : 'chip-inactive'}`}>BM25: {useHybrid ? 'ON' : 'OFF'}</div>
            <div className={`chip ${useRerank ? 'chip-active' : 'chip-inactive'}`}>Reranking: {useRerank ? 'ON' : 'OFF'}</div>
          </div>
          <button className="clear-btn" onClick={clearIndex} title="Wipe all indexed documents">
            🗑️ Clear Index
          </button>
        </div>
      </aside>

      {/* Sidebar Toggle */}
      <button className="sidebar-toggle-btn" onClick={() => setSidebarOpen(v => !v)}>
        {sidebarOpen ? <ChevronLeft size={18} /> : <ChevronRight size={18} />}
      </button>

      {/* ═══ MAIN CHAT ═══ */}
      <main className="chat-container">
        <div className={`messages-area ${clearingChat ? 'chat-clearing' : ''}`}>
          {chatHistory.length === 0 && (
            <div className="empty-state">
              <FileText size={52} className="icon-subtle" />
              <h2>Ask a Legal Question.</h2>
              <p>Upload a contract on the left and start investigating.</p>
            </div>
          )}
          {chatHistory.map((msg, i) => (
            <div
              key={i}
              className={`message-bubble ${msg.role}`}
              style={clearingChat ? { animationDelay: `${i * 40}ms` } : {}}
            >
              {msg.role === 'bot'
                ? <ReactMarkdown>{msg.content}</ReactMarkdown>
                : msg.content
              }
            </div>
          ))}
          {loading && (
            <div className="message-bubble bot typing-indicator">
              <span /><span /><span />
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={handleAsk} className="input-area glass-panel">
          <input
            type="text"
            placeholder="Ask about clauses, penalties, or termination..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          {chatHistory.length > 0 && (
            <button
              type="button"
              className={`clear-chat-btn ${clearingChat ? 'clearing' : ''}`}
              onClick={handleClearChat}
              disabled={clearingChat}
              title="Clear chat history"
            >
              <Trash2 size={15} />
              <span>Clear</span>
            </button>
          )}
          <button type="submit" disabled={loading || !query.trim()} className="send-btn">
            <Send size={18} />
          </button>
        </form>
      </main>

      {/* Inspector Toggle */}
      <button className="inspector-toggle-btn" onClick={() => setInspectorOpen(v => !v)}>
        {inspectorOpen ? <PanelRightClose size={18} /> : <PanelRightOpen size={18} />}
      </button>

      {/* ═══ INSPECTOR PANEL ═══ */}
      <aside className={`inspector-panel glass-panel ${inspectorOpen ? 'inspector-visible' : 'inspector-hidden'}`}>
        <div className="inspector-header">
          <h3><Search size={16} /> Source Inspection</h3>
          <button className="icon-btn" onClick={() => setInspectorOpen(false)} title="Close">
            <X size={16} />
          </button>
        </div>
        <p className="inspector-desc">Click any chunk to view full contents.</p>

        <div className="evidence-list">
          {evidence.length === 0
            ? <p className="status-text" style={{ color: 'var(--text-secondary)' }}>No chunks fetched yet. Ask a question.</p>
            : evidence.map((chunk, i) => (
              <div
                key={i}
                className="evidence-card clickable-card"
                onClick={() => setSelectedChunk({ chunk, index: i })}
                title="Click to expand"
              >
                <div className="evidence-meta">
                  <span className="badge">Chunk {i + 1}</span>
                  <span className="source-name">{chunk.source} · Pg {chunk.page}</span>
                  <ExternalLink size={11} className="card-expand-icon" />
                </div>
                <p className="evidence-text">"{chunk.text}"</p>
                <div className="score-bar">
                  <div className="score-fill" style={{ width: `${Math.min(100, Math.max(0, 100 - chunk.distance * 100))}%` }} />
                </div>
                <small className="distance-label">Distance: {chunk.distance.toFixed(4)}</small>
              </div>
            ))
          }
        </div>
      </aside>

      {/* ═══ CHUNK MODAL ═══ */}
      {selectedChunk && (
        <ChunkModal
          chunk={selectedChunk.chunk}
          index={selectedChunk.index}
          onClose={() => setSelectedChunk(null)}
        />
      )}
    </div>
  );
}

export default App;
