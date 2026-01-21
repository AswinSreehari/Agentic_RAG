import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { FaPaperclip, FaArrowUp, FaTrash } from 'react-icons/fa';
import './ChatInterface.css';

const API_BASE = "http://localhost:8000";

const ChatInterface = () => {
    const [messages, setMessages] = useState(() => {
        try {
            const saved = localStorage.getItem('chat_history');
            const parsed = saved ? JSON.parse(saved) : [];
            const valid = parsed.filter(m => m && m.content);
            return valid.length > 0 ? valid : [
                { role: 'assistant', content: "Hello! I'm your RAG Agent. Upload a document to get started." }
            ];
        } catch (e) {
            console.error("Failed to load history", e);
            return [{ role: 'assistant', content: "Hello! I'm your RAG Agent. Upload a document to get started." }];
        }
    });
    
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [currentStatus, setCurrentStatus] = useState("");
    const [isThinking, setIsThinking] = useState(false);
    const fileInputRef = useRef(null);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
        localStorage.setItem('chat_history', JSON.stringify(messages));
    }, [messages]);

    const handleSend = async () => {
        if (!input.trim()) return;

        const userMsg = { role: 'user', content: input };
        setMessages(prev => [...prev, userMsg]);
        setInput("");
        setIsLoading(true);
        setIsLoading(true);
        setIsThinking(true);
        setCurrentStatus("Starting...");

        try {
            const history = messages
                .filter(m => m.role !== 'system' && m.content)  
                .map(m => ({ role: m.role, content: m.content }));

            const response = await fetch(`${API_BASE}/chat_stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: userMsg.content, history })
            });

            if (!response.body) throw new Error("ReadableStream not supported");

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let finalRes = "";
            let finalSrc = [];

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const data = JSON.parse(line);
                        if (data.type === 'thinking') {
                            let status = data.content;
                            if (status.includes("ACTION:")) status = "Searching documents..."; 
                            else if (status.includes("OBSERVATION:")) status = "Found results...";  
                            else if (status.includes("FINAL_ANSWER:")) status = "Answering...";
                             
                            if (data.content.includes("search_documents")) setCurrentStatus("Searching documents...");
                            else if (data.content.includes("OBSERVATION")) setCurrentStatus("Found results...");
                            else setCurrentStatus("Analyzing...");

                        } else if (data.type === 'final') {
                            finalRes = data.response;
                            finalSrc = data.sources;
                            setCurrentStatus("Finished");
                        } else if (data.type === 'error') {
                            console.error("Stream Error:", data.content);
                        }
                    } catch (e) {
                        console.error("Error parsing JSON:", e);
                    }
                }
            }

            const aiMsg = {
                role: 'assistant',
                content: finalRes || "I couldn't generate a response. Please try again.",
                sources: finalSrc
            };
            setMessages(prev => [...prev, aiMsg]);
        } catch (error) {
            setMessages(prev => [...prev, { role: 'assistant', content: "Sorry, something went wrong connecting to the agent." }]);
            console.error(error);
        } finally {
            setIsLoading(false);
            setIsThinking(false);
        }
    };

    const handleClearChat = () => {
        if (window.confirm("Are you sure you want to clear the chat history?")) {
            const initialmsg = [{ role: 'assistant', content: "Hello! I'm your RAG Agent. Upload a document to get started." }];
            setMessages(initialmsg);
            localStorage.setItem('chat_history', JSON.stringify(initialmsg));
        }
    };

    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append("file", file);

        setMessages(prev => [...prev, { role: 'system', content: `Uploading ${file.name}...` }]);

        try {
            const res = await axios.post(`${API_BASE}/upload`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            setMessages(prev => [...prev, { role: 'system', content: `✅ ${res.data.message}` }]);
        } catch (error) {
            setMessages(prev => [...prev, { role: 'system', content: `❌ Error uploading ${file.name}` }]);
        }
        e.target.value = null;
    };

    return (
        <div className="chat-container">
            <div className="chat-header">
                <div className="avatar-circle">AI</div>
                <div className="header-info">
                    <h2>RAG Agent</h2>
                    <span className="status">Online</span>
                </div>
                <button className="icon-btn" onClick={handleClearChat} title="Clear Chat" style={{ marginLeft: 'auto' }}>
                    <FaTrash />
                </button>
            </div>

            <div className="messages-area">
                {messages.map((msg, idx) => (
                    <div key={idx} className={`message-row ${msg.role}`}>
                        {msg.role !== 'system' && <div className="message-sender">{msg.role === 'user' ? 'You' : 'Agent'}</div>}
                        <div className={`message-bubble ${msg.role}`}>
                            <div className="content">{msg.role === 'system' ? <span className="system-msg">{msg.content}</span> : msg.content}</div>

                            {msg.sources && msg.sources.length > 0 && (
                                <div className="sources-container">
                                    <div className="sources-title">Sources:</div>
                                    {msg.sources.map((src, sIdx) => (
                                        <div key={sIdx} className="source-item">
                                            <div className="source-meta">📄 {src.filename} (Page {src.page})</div>
                                            {src.content && <div className="source-content">"{src.content}"</div>}
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                ))}
                {isLoading && (
                    <div className="message-row assistant">
                        <div className="message-bubble assistant typing">
                            <span>.</span><span>.</span><span>.</span>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            <div className="input-area">
                {isThinking && currentStatus && (
                    <div className="thinking-status" style={{
                        marginBottom: '10px',
                        padding: '0 10px',
                        color: '#9ca3af',
                        fontSize: '0.9rem',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px'
                    }}>
                        <div className="spinner" style={{
                            width: '12px',
                            height: '12px',
                            border: '2px solid rgba(156, 163, 175, 0.3)',
                            borderTop: '2px solid #9ca3af',
                            borderRadius: '50%',
                            animation: 'spin 1s linear infinite'
                        }}></div>
                        <span>{currentStatus}</span>
                        <style>{`
                            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
                        `}</style>
                    </div>
                )}
                <div className="input-row">
                    <input
                        type="file"
                        ref={fileInputRef}
                        onChange={handleFileUpload}
                        style={{ display: 'none' }}
                    />
                    <button className="icon-btn" onClick={() => fileInputRef.current.click()} title="Upload File">
                        <FaPaperclip />
                    </button>
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                        placeholder="Type your message here..."
                    />
                    <button className="send-btn" onClick={handleSend} disabled={!input.trim()}>
                        <FaArrowUp />
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ChatInterface;
