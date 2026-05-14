import React, { useState, useRef, useEffect } from 'react'
import { Send, Terminal, Bot, User, Loader2, Sparkles, AlertCircle } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

function App() {
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState([])
  const [thoughtProcess, setThoughtProcess] = useState([])
  const [summary, setSummary] = useState('')
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState('offline')
  
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    // Check backend status
    fetch('http://localhost:8000/status')
      .then(res => res.json())
      .then(data => setStatus('online'))
      .catch(() => setStatus('offline'))
  }, [])

  const handleSend = async () => {
    if (!input.trim() || loading) return

    const userMessage = { role: 'user', content: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setLoading(true)
    setThoughtProcess([]) // Clear previous thoughts

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: input,
          history: messages,
          summary: summary
        })
      })

      if (!response.ok) throw new Error('API Error')

      const data = await response.json()
      
      setMessages(data.history)
      setThoughtProcess(data.thought_process)
      setSummary(data.summary)
    } catch (error) {
      console.error(error)
      setMessages(prev => [...prev, { role: 'assistant', content: '⚠️ Error: Could not connect to the agent. Make sure the backend is running.' }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      {/* Sidebar: Thought Trace */}
      <aside className="sidebar">
        <div className="sidebar-title">
          <Terminal size={14} style={{ marginRight: '8px' }} />
          Agent Thought Trace
        </div>
        
        {thoughtProcess.length === 0 && !loading && (
          <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem', opacity: 0.5 }}>
            Agent reasoning will appear here...
          </div>
        )}

        {loading && (
          <div className="thought-step decision">
            <Loader2 size={16} className="animate-spin" style={{ marginBottom: '8px', color: 'var(--accent-blue)' }} />
            Processing query... Analyzing domain and intent.
          </div>
        )}

        {thoughtProcess.map((thought, i) => (
          <div key={i} className={`thought-step ${thought.toLowerCase().includes('decision') ? 'decision' : 'retrieve'}`}>
            {thought}
          </div>
        ))}

        {summary && (
          <div style={{ marginTop: 'auto', paddingTop: '2rem' }}>
            <div className="sidebar-title">Semantic Memory</div>
            <div className="thought-step" style={{ background: 'rgba(0,0,0,0.2)', fontSize: '0.8rem' }}>
              {summary}
            </div>
          </div>
        )}
      </aside>

      {/* Main Chat */}
      <main className="chat-area">
        <header className="chat-header">
          <h1>Agentic RAG Assistant</h1>
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ 
              width: '8px', 
              height: '8px', 
              borderRadius: '50%', 
              background: status === 'online' ? '#10b981' : '#ef4444' 
            }} />
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Backend: {status}
            </span>
          </div>
        </header>

        <div className="messages-container">
          {messages.length === 0 && (
            <div style={{ 
              display: 'flex', 
              flexDirection: 'column', 
              alignItems: 'center', 
              justifyContent: 'center', 
              height: '100%', 
              color: 'var(--text-muted)',
              textAlign: 'center'
            }}>
              <Sparkles size={48} style={{ marginBottom: '1rem', opacity: 0.2 }} />
              <h3>Welcome to the Agentic Research Assistant</h3>
              <p style={{ maxWidth: '400px', fontSize: '0.875rem', marginTop: '0.5rem' }}>
                Ask me about recent AI papers from arXiv. I use multi-step reasoning and RAG-Fusion to give you the best answers.
              </p>
            </div>
          )}
          
          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role}`}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px', opacity: 0.6, fontSize: '0.75rem' }}>
                {msg.role === 'user' ? <User size={12} /> : <Bot size={12} />}
                {msg.role === 'user' ? 'You' : 'Agent'}
              </div>
              <div className="message-content">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {msg.content}
                </ReactMarkdown>
              </div>
            </div>
          ))}
          {loading && (
            <div className="message assistant" style={{ opacity: 0.7 }}>
              <div style={{ display: 'flex', gap: '4px' }}>
                <span className="dot">.</span>
                <span className="dot">.</span>
                <span className="dot">.</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-container">
          <div className="input-wrapper">
            <input 
              type="text" 
              placeholder="Ask about AI research..." 
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            />
            <button 
              className="send-btn" 
              onClick={handleSend}
              disabled={loading || !input.trim()}
            >
              {loading ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
            </button>
          </div>
        </div>
      </main>

      <style dangerouslySetInnerHTML={{ __html: `
        .animate-spin { animation: spin 1s linear infinite; }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        .dot { animation: pulse 1.5s infinite; }
        .dot:nth-child(2) { animation-delay: 0.2s; }
        .dot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes pulse { 0%, 100% { opacity: 0.2; } 50% { opacity: 1; } }
      `}} />
    </div>
  )
}

export default App
