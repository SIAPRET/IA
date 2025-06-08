const { useState, useEffect, useRef } = React;

const ChatApp = () => {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  
  // Reemplaza con tu IP local
  const BACKEND_URL = 'http://localhost:5000/analyze-text';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!inputText.trim()) return;

    // Agregar mensaje del usuario
    const userMessage = {
      id: Date.now(),
      text: inputText,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      // Enviar al backend
      const response = await axios.post(BACKEND_URL, {
        text: inputText
      });

      // Procesar respuesta
      const { prediction, probability, symptoms_detected } = response.data;
      const confidence = Math.round(probability * 100);
      
      let aiMessage;
      if (prediction) {
        aiMessage = {
          id: Date.now() + 1,
          text: `⚠️ <strong>Posible caso de tuberculosis (${confidence}% de confianza)</strong><br><br>
                ${symptoms_detected?.length ? '🔍 <strong>Síntomas detectados:</strong><br>' + symptoms_detected.map(s => `• ${s}`).join('<br>') + '<br><br>' : ''}
                💡 <strong>Recomendación:</strong> Consulta con un especialista para una evaluación más detallada.`,
          sender: 'ai',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          isHTML: true
        };
      } else {
        aiMessage = {
          id: Date.now() + 1,
          text: `✅ <strong>Baja probabilidad de tuberculosis (${confidence}% de confianza)</strong><br><br>
                ${symptoms_detected?.length ? '🔍 <strong>Síntomas detectados:</strong><br>' + symptoms_detected.map(s => `• ${s}`).join('<br>') + '<br><br>' : ''}
                💡 <strong>Recomendación:</strong> Si los síntomas persisten, consulta con un médico.`,
          sender: 'ai',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          isHTML: true
        };
      }

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        text: '❌ Lo siento, hubo un error al procesar tu consulta. Por favor inténtalo de nuevo.',
        sender: 'ai',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }]);
      console.error('Error al conectar con el backend:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="app-container">
      <div className="header">
        <h1><i className="fas fa-lungs-virus"></i> Asistente de TB</h1>
        <p>Describe tus síntomas para recibir una evaluación</p>
      </div>

      <div className="chat-container">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`message-bubble ${message.sender === 'user' ? 'user-bubble' : 'ai-bubble'}`}
          >
            <div className={`message-text ${message.sender === 'user' ? 'user-text' : 'ai-text'}`}>
              {message.isHTML ? <div dangerouslySetInnerHTML={{ __html: message.text }} /> : message.text}
            </div>
            <div className="timestamp">{message.timestamp}</div>
          </div>
        ))}
        
        {isLoading && (
          <div className="message-bubble ai-bubble">
            <div className="loading-indicator">
              <div className="spinner"></div>
              <span>Analizando síntomas...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-container">
        <textarea
          className="text-input"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Escribe tus síntomas aquí..."
          rows="3"
        />
        <button
          className="send-button"
          onClick={handleSend}
          disabled={isLoading || !inputText.trim()}
        >
          <i className="fas fa-paper-plane"></i> Enviar
        </button>
      </div>
    </div>
  );
};

// Estilos mejorados
const style = document.createElement('style');
style.textContent = `
  .app-container {
    display: flex;
    flex-direction: column;
    height: 100vh;
    max-width: 800px;
    margin: 0 auto;
    background-color: #ffffff;
    box-shadow: 0 0 20px rgba(0,0,0,0.1);
    border-radius: 12px;
    overflow: hidden;
  }
  
  .header {
    padding: 20px;
    background-color: #3498db;
    color: white;
    text-align: center;
  }
  
  .header h1 {
    margin: 0;
    font-size: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
  }
  
  .header p {
    margin: 8px 0 0;
    font-size: 14px;
    opacity: 0.9;
  }
  
  .chat-container {
    flex: 1;
    padding: 20px;
    overflow-y: auto;
    background-color: #f5f7fb;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  
  .message-bubble {
    max-width: 85%;
    padding: 12px 16px;
    border-radius: 18px;
    position: relative;
    line-height: 1.5;
    animation: fadeIn 0.3s ease-out;
  }
  
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }
  
  .user-bubble {
    align-self: flex-end;
    background-color: #3498db;
    color: white;
    border-bottom-right-radius: 4px;
  }
  
  .ai-bubble {
    align-self: flex-start;
    background-color: #ffffff;
    border: 1px solid #e1e5eb;
    border-bottom-left-radius: 4px;
  }
  
  .message-text {
    font-size: 15px;
  }
  
  .user-text {
    color: white;
  }
  
  .ai-text {
    color: #2c3e50;
  }
  
  .timestamp {
    font-size: 11px;
    color: #7f8c8d;
    margin-top: 8px;
    text-align: right;
  }
  
  .input-container {
    display: flex;
    padding: 15px;
    background-color: #ffffff;
    border-top: 1px solid #e1e5eb;
    gap: 10px;
  }
  
  .text-input {
    flex: 1;
    padding: 12px 15px;
    border: 1px solid #e1e5eb;
    border-radius: 20px;
    resize: none;
    font-family: inherit;
    font-size: 15px;
    min-height: 50px;
    max-height: 120px;
    transition: border-color 0.3s;
  }
  
  .text-input:focus {
    outline: none;
    border-color: #3498db;
    box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
  }
  
  .send-button {
    padding: 0 20px;
    background-color: #3498db;
    color: white;
    border: none;
    border-radius: 20px;
    font-weight: 500;
    cursor: pointer;
    height: 50px;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: all 0.3s;
  }
  
  .send-button:hover {
    background-color: #2980b9;
    transform: translateY(-1px);
  }
  
  .send-button:active {
    transform: translateY(0);
  }
  
  .send-button:disabled {
    background-color: #bdc3c7;
    cursor: not-allowed;
    transform: none;
  }
  
  .loading-indicator {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #7f8c8d;
  }
  
  .spinner {
    border: 3px solid rgba(0,0,0,0.1);
    border-radius: 50%;
    border-top: 3px solid #3498db;
    width: 18px;
    height: 18px;
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }

  @media (max-width: 600px) {
    .app-container {
      border-radius: 0;
      height: 100vh;
    }
    
    .message-bubble {
      max-width: 90%;
    }
    
    .input-container {
      padding: 12px;
    }
    
    .send-button {
      padding: 0 15px;
    }
  }
`;
document.head.appendChild(style);

// Renderizar la aplicación
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <ChatApp />
  </React.StrictMode>
);