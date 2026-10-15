import React, { useState, useEffect } from 'react';
import './App.css';
import logoIfms from './logoIF.png';

function App() {
  const [rnaCode, setRnaCode] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  // Carrega o histórico do LocalStorage assim que a página abre
  useEffect(() => {
    const savedHistory = localStorage.getItem('mirtrons_history');
    if (savedHistory) {
      setHistory(JSON.parse(savedHistory));
    }
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!rnaCode.trim()) return alert("Por favor, digite o código de RNA.");

    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/classify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ rna_code: rnaCode }),
      });

      const data = await response.json();
      setResult(data);

      // Cria o novo item para o histórico
      const newItem = {
        id: Date.now(),
        date: new Date().toLocaleString('pt-BR'),
        input: rnaCode,
        output: data.classification || JSON.stringify(data) // Ajuste conforme a resposta do seu Python
      };

      // Atualiza o estado e salva no LocalStorage do navegador
      const updatedHistory = [newItem, ...history];
      setHistory(updatedHistory);
      localStorage.setItem('mirtrons_history', JSON.stringify(updatedHistory));

    } catch (error) {
      console.error("Erro ao enviar dados para o backend:", error);
      alert("Erro ao conectar com o servidor local em Python.");
    } finally {
      setLoading(false);
    }
  };

  // Função para resetar todo o progresso e histórico
  const handleReset = () => {
    if (window.confirm("Tem certeza que deseja apagar todo o histórico e limpar a tela?")) {
      setRnaCode('');
      setResult(null);
      setHistory([]);
      localStorage.removeItem('mirtrons_history');
    }
  };

  return (
    
    <div className="container">
      {/* Logo do IFMS local */}
      {/*<header className="header">
        <img 
          src={logoIfms} 
          alt="Logo IFMS" 
          className="logo-ifms"
        />
      </header>*/}

      <main className="main-content">
        <h1 className="title">Ferramenta de classificação de mirtrons</h1>
        
        <form onSubmit={handleSubmit} className="search-box">
          <textarea
            className="input-rna"
            placeholder="Digite o código de RNA"
            value={rnaCode}
            onChange={(e) => setRnaCode(e.target.value)}
            rows={4}
          />
          
          <div className="button-group">
            <button type="submit" className="btn-submit" disabled={loading}>
              {loading ? 'Classificando...' : 'Classificar'}
            </button>
            
            {(rnaCode || result || history.length > 0) && (
              <button type="button" onClick={handleReset} className="btn-reset">
                Resetar Progresso
              </button>
            )}
          </div>
        </form>

        {/* Resultado Atual */}
        {result && (
          <div className="result-box">
            <h3>Resultado Atual:</h3>
            <p>{JSON.stringify(result)}</p>
          </div>
        )}

        {/* Seção de Histórico Local */}
        {history.length > 0 && (
          <div className="history-section">
            <h2 className="history-title">Histórico de Consultas Local</h2>
            <div className="history-list">
              {history.map((item) => (
                <div key={item.id} className="history-item">
                  <span className="history-date">{item.date}</span>
                  <div className="history-details">
                    <p><strong>RNA:</strong> <code className="code-text">{item.input}</code></p>
                    <p><strong>Resultado:</strong> {item.output}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
