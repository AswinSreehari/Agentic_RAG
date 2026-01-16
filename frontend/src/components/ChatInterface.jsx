import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { FaPaperclip, FaArrowUp } from 'react-icons/fa';
import './ChatInterface.css';

const API_BASE = "http://localhost:8000";

const ChatInterface = () => {
    const [messages, setMessages] = useState([
        { role: 'assistant', content: "Hello! I'm your RAG Agent. Upload a document to get started." }
    ]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const fileInputRef = useRef(null);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async () => {
        if (!input.trim()) return;

        const userMsg = { role: 'user', content: input };
        setMessages(prev => [...prev, userMsg]);
        setInput("");
        setIsLoading(true);

        try {
            const history = messages
                .filter(m => m.role !== 'system')
                .map(m => ({ role: m.role, content: m.content }));

            const response = await axios.post(`${API_BASE}/chat`, {
                message: userMsg.content,
                history: history
            });

            const aiMsg = {
                role: 'assistant',
                content: response.data.response,
                sources: response.data.sources
            };
            setMessages(prev => [...prev, aiMsg]);
        } catch (error) {
            setMessages(prev => [...prev, { role: 'assistant', content: "Sorry, something went wrong connecting to the agent." }]);
            console.error(error);
        } finally {
            setIsLoading(false);
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
        // clear input
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
    );
};

export default ChatInterface;
