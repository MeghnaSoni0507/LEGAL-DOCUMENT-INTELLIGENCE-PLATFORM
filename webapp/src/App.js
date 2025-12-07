import React, { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [extractedData, setExtractedData] = useState(null);

  const [summaries, setSummaries] = useState([]);
  const [sections, setSections] = useState([]);
  const [entities, setEntities] = useState([]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [searching, setSearching] = useState(false);

  const [question, setQuestion] = useState("");
  const [aiAnswer, setAiAnswer] = useState("");
  const [aiLoading, setAiLoading] = useState(false);

  const [anomalies, setAnomalies] = useState([]);
  const [anomalyFeedback, setAnomalyFeedback] = useState("");
  const [detecting, setDetecting] = useState(false);

  const BACKEND_URL = "http://127.0.0.1:5000";


  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected && selected.type !== "application/pdf") {
      setError("Please upload a valid PDF");
      return;
    }
    setError("");
    setFile(selected);
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a PDF first!");
      return;
    }

    setLoading(true);
    setError("");
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${BACKEND_URL}/extract`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("File processing failed");

      const data = await res.json();

      setExtractedData(data);
      setSummaries(data.summaries || []);
      setSections(data.sections || []);
      setEntities(data.entities || []);

      setSearchResults([]);
      setAiAnswer("");
      setAnomalies([]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };


  const handleSearch = async () => {
    if (!searchQuery.trim()) return alert("Enter a query");
    if (!extractedData) return alert("Upload a document first");

    setSearching(true);
    setError("");

    try {
      const res = await fetch(`${BACKEND_URL}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: searchQuery,
          text: extractedData.full_cleaned_text,
        }),
      });

      if (!res.ok) throw new Error("Search failed");

      const data = await res.json();
      setSearchResults(data.results || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setSearching(false);
    }
  };


  const handleAskAI = async () => {
    if (!question.trim()) return alert("Enter a question");
    if (!extractedData) return alert("Upload a document first");

    setAiLoading(true);
    setError("");

    try {
      const res = await fetch(`${BACKEND_URL}/ask-ai`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          document_text: extractedData.full_cleaned_text,
        }),
      });

      const data = await res.json();

      if (!res.ok) throw new Error(data.error || "AI request failed");

      setAiAnswer(data.answer || "No answer returned");
    } catch (err) {
      setError(err.message);
      setAiAnswer("");
    } finally {
      setAiLoading(false);
    }
  };


  const handleDetectAnomalies = async () => {
    if (!extractedData) return alert("Upload a document first");

    setDetecting(true);
    setError("");
    setAnomalies([]);
    setAnomalyFeedback("");

    try {
      const res = await fetch(`${BACKEND_URL}/detect-anomalies`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document_text:
            extractedData.full_cleaned_text || extractedData.full_raw_text,
        }),
      });

      const data = await res.json();

      if (!res.ok) throw new Error(data.error || "Anomaly detection failed");

      setAnomalies(data.found_clauses || []);
      setAnomalyFeedback(data.ai_feedback || "");
    } catch (err) {
      setError(err.message);
    } finally {
      setDetecting(false);
    }
  };


  return (
    <div className="App">
      <div className="container">
        <h1>⚖️ Legal Document Intelligence</h1>
        <p className="subtitle">Upload a contract and interact with AI</p>

        {/* Upload */}
        <div className="upload-section">
          <input type="file" accept="application/pdf" onChange={handleFileChange} />

          <button
            className="btn-primary"
            disabled={loading || !file}
            onClick={handleUpload}
          >
            {loading ? "Processing..." : "📤 Extract Document"}
          </button>
        </div>

        {error && <div className="error-box">{error}</div>}

        {extractedData && (
          <>
            <div className="info-box">
              <p><strong>📄 File:</strong> {extractedData.filename}</p>
              <p><strong>📃 Pages:</strong> {extractedData.pages}</p>
              {extractedData.ocr_pages?.length > 0 && (
                <p><strong>OCR Used On:</strong> {extractedData.ocr_pages.join(", ")}</p>
              )}
            </div>

            {/* Search */}
            <div className="search-box">
              <h3>🔎 Search Document</h3>

              <div className="search-controls">
                <input
                  type="text"
                  placeholder="Search for terms or clauses..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                />
                <button className="btn-secondary" onClick={handleSearch}>
                  {searching ? "Searching..." : "Search"}
                </button>
              </div>

              {searchResults.length > 0 && (
                <div className="search-results">
                  {searchResults.map((s, i) => (
                    <p key={i} className="result-item">{s}</p>
                  ))}
                </div>
              )}
            </div>

            {/* AI Box */}
            <div className="ai-box">
              <h3>💬 GenAI Legal Assistant</h3>
              <input
                type="text"
                placeholder="Ask: What are the termination terms?"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleAskAI()}
              />

              <button className="btn-ai" onClick={handleAskAI}>
                {aiLoading ? "Thinking..." : "🤖 Ask AI"}
              </button>

              {aiAnswer && (
                <div className="ai-answer">
                  <strong>Answer:</strong>
                  <p>{aiAnswer}</p>
                </div>
              )}
            </div>

            {/* Anomaly Detection */}
            <div className="anomaly-box">
              <h3>⚠️ Anomaly / Risk Detection</h3>

              <button className="btn-warning" onClick={handleDetectAnomalies}>
                {detecting ? "Scanning..." : "🕵️ Detect Risks"}
              </button>

              {anomalies.length > 0 && (
                <div className="anomaly-results">
                  {anomalies.map((a, i) => (
                    <div className="anomaly-item" key={i}>
                      <strong>{a.pattern.toUpperCase()}</strong>
                      <p>{a.snippet}</p>
                    </div>
                  ))}

                  {anomalyFeedback && (
                    <div className="ai-feedback">
                      <h4>AI Summary:</h4>
                      <p>{anomalyFeedback}</p>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Entities */}
            <div className="entities-box">
              <h3>🧩 Extracted Entities</h3>

              {entities.length > 0 ? (
                <div className="entity-list">
                  {entities.map((ent, i) => (
                    <span key={i} className="entity-tag">
                      {ent.text} <em>({ent.label})</em>
                    </span>
                  ))}
                </div>
              ) : (
                <p>No entities found</p>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default App;
