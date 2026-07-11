import { useState, useRef } from 'react';
import './index.css';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    if (!file.type.match('image.*')) {
      setError('Please select an image file (JPEG, PNG, etc).');
      return;
    }
    setFile(file);
    setError(null);
    setResult(null);
    const objectUrl = URL.createObjectURL(file);
    setPreview(objectUrl);
  };

  const onScanClick = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await fetch('http://localhost:3001/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data.prediction);
    } catch (err) {
      console.error(err);
      setError('Failed to connect to the prediction server. Make sure the Node.js backend is running on port 3001.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>Pneumonia Detection</h1>
        <p className="subtitle">Powered by an incredible Custom-built Convolutional Neural Network from Scratch</p>
      </header>

      <main>
        <div className="upload-card">
          {!preview ? (
            <div
              className={`drop-zone ${dragActive ? 'active' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current.click()}
            >
              <svg className="upload-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path>
              </svg>
              <h3>Drag & drop an X-Ray image</h3>
              <p style={{ marginTop: '0.5rem', color: 'var(--text-muted)' }}>Or click to browse your files</p>
              <input
                ref={fileInputRef}
                type="file"
                style={{ display: 'none' }}
                onChange={handleChange}
                accept="image/*"
              />
            </div>
          ) : (
            <div className="preview-container">
              <img src={preview} alt="X-ray preview" className="image-preview" />
              <div style={{ marginBottom: '1.5rem' }}>
                <button
                  style={{ background: 'transparent', border: '1px solid var(--border)', padding: '0.5rem 1rem', borderRadius: '0.5rem', cursor: 'pointer', marginRight: '1rem' }}
                  onClick={() => { setPreview(null); setFile(null); setResult(null); setError(null); }}
                  disabled={loading}
                >
                  Choose Different Image
                </button>
              </div>

              <button
                className="btn"
                onClick={onScanClick}
                disabled={loading}
              >
                {loading ? (
                  <><span className="loading-spinner"></span> Scanning Image...</>
                ) : (
                  'Analyze X-Ray'
                )}
              </button>
            </div>
          )}

          {error && <div className="error-message">{error}</div>}

          {result && (
            <div className={`result-card ${typeof result === 'string' ? result.toLowerCase() : result.Pneumonia > 0.5 ? 'pneumonia' : 'normal'}`}>
              <div className="result-title">Detection Result</div>
              <div className="result-confidence">
                {typeof result === 'string'
                  ? result
                  : (result.Pneumonia > 0.5 ? `Pneumonia Detected (${(result.Pneumonia * 100).toFixed(1)}%)` : `Normal Scan (${(result.Normal * 100).toFixed(1)}%)`)
                }
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
