// src/App.js
import React, { useEffect, useRef, useState } from "react";
import "./App.css";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:5000";
const POLL_INTERVAL_MS = 1000; // 1s
const POLL_TIMEOUT_MS = 2 * 60 * 1000; // 2 minutes

export default function App() {
  // upload / job states
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [statusText, setStatusText] = useState("");
  const [progress, setProgress] = useState({ processed: 0, total_to_process: 0 });
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  // feature states
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [aiQuestion, setAiQuestion] = useState("");
  const [aiAnswer, setAiAnswer] = useState(null);
  const [anomalyResult, setAnomalyResult] = useState(null);

  // utility
  const pollTimerRef = useRef(null);
  const pollStartRef = useRef(null);

  useEffect(() => {
    return () => stopPolling();
    // eslint-disable-next-line
  }, []);

  function stopPolling() {
    if (pollTimerRef.current) {
      clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }

  const handleFileChange = (e) => {
    setError("");
    setResult(null);
    setJobId(null);
    const f = e.target.files && e.target.files[0];
    if (!f) {
      setFile(null);
      return;
    }
    if (f.type !== "application/pdf" && !f.name.toLowerCase().endsWith(".pdf")) {
      setError("Please select a PDF file.");
      setFile(null);
      return;
    }
    setFile(f);
  };

  const handleUpload = async () => {
    setError("");
    setAiAnswer(null);
    setAnomalyResult(null);
    setSearchResults([]);
    setResult(null);

    if (!file) {
      setError("Choose a PDF first.");
      return;
    }

    setUploading(true);
    setProcessing(true);
    setStatusText("Uploading PDF...");
    setProgress({ processed: 0, total_to_process: 0 });
    setJobId(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const resp = await fetch(`${BACKEND_URL}/extract`, { method: "POST", body: formData });
      if (resp.status === 202) {
        const data = await resp.json();
        const jid = data.job_id;
        setJobId(jid);
        setStatusText("Queued — waiting for worker");
        pollStartRef.current = Date.now();
        pollStatus(jid);
      } else {
        const text = await resp.text();
        setError(`Upload failed: ${resp.status} ${resp.statusText} — ${text}`);
        setProcessing(false);
      }
    } catch (err) {
      setError(`Upload error: ${err.message}`);
      setProcessing(false);
    } finally {
      setUploading(false);
    }
  };

  const pollStatus = async (jid) => {
    stopPolling();

    const elapsed = Date.now() - (pollStartRef.current || 0);
    if (elapsed > POLL_TIMEOUT_MS) {
      setError("Processing timed out. Try a smaller file or increase timeout.");
      setProcessing(false);
      return;
    }

    try {
      const resp = await fetch(`${BACKEND_URL}/status/${jid}`, { method: "GET" });
      if (resp.status === 404) {
        setError("Job not found on server.");
        setProcessing(false);
        return;
      }
      const data = await resp.json();

      if (data.status === "completed" || data.status === "done") {
        setStatusText("Completed");
        const res = data.result || {};
        const normalized = {
          processed: res.pages_processed ?? data.processed_pages ?? 0,
          total_to_process: res.pages ?? data.total_pages ?? data.pages_to_process ?? 0,
        };
        setProgress(normalized);
        setResult(res);
        setProcessing(false);
        stopPolling();
        return;
      }

      if (data.status === "error") {
        setError(`Processing error: ${data.error || "unknown"}`);
        setProcessing(false);
        stopPolling();
        return;
      }

      setStatusText(data.status || "processing");

      const norm = {
        processed:
          data.processed_pages ??
          (typeof data.progress === "number" && typeof data.total_pages === "number"
            ? Math.round((data.progress / 100) * data.total_pages)
            : data.processed ?? 0),
        total_to_process: data.total_pages ?? data.pages_to_process ?? 0,
      };
      setProgress(norm);

      pollTimerRef.current = setTimeout(() => pollStatus(jid), POLL_INTERVAL_MS);
    } catch (err) {
      console.error("Poll error:", err);
      setStatusText("Waiting to reconnect...");
      pollTimerRef.current = setTimeout(() => pollStatus(jid), POLL_INTERVAL_MS);
    }
  };

  const cancelJob = () => {
    stopPolling();
    setProcessing(false);
    setStatusText("Cancelled");
    setJobId(null);
    setProgress({ processed: 0, total_to_process: 0 });
  };

  // ---------- Search ----------
  const handleSearch = async (e) => {
    // important: prevent normal form submit which causes page reload/top scroll
    if (e && e.preventDefault) e.preventDefault();

    setError("");
    setSearchResults([]);

    if (!searchQuery.trim()) {
      setError("Please enter a search query.");
      return;
    }

    if (!result || !result.full_cleaned_text) {
      setError("Extraction not done or no text available.");
      return;
    }

    try {
      console.log("[search] sending", { query: searchQuery });
      const resp = await fetch(`${BACKEND_URL}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: searchQuery,
          full_cleaned_text: result.full_cleaned_text
        })
      });

      const text = await resp.text();
      let data;
      try { data = JSON.parse(text); } catch { data = { results: [] }; }

      console.log("[search] resp", resp.status, data);

      if (!resp.ok) {
        setError(data.error || `Search failed: ${resp.status}`);
        return;
      }

      // normalize results: accept array of strings or objects
      const raw = data.results ?? [];
      const normalized = raw.map((r) => {
        if (typeof r === "string") return { text: r };
        if (r && typeof r === "object") return r;
        return { text: String(r) };
      });

      setSearchResults(normalized);
    } catch (e) {
      console.error("[search] error", e);
      setError("Search failed: " + e.message);
    }
  };

  // ---------- AI Q&A ----------
  const handleAskAI = async (e) => {
    if (e && e.preventDefault) e.preventDefault();

    setAiAnswer(null);
    setError("");

    if (!aiQuestion.trim()) {
      setError("Please type a question.");
      return;
    }

    if (!result || !result.full_cleaned_text) {
      setError("Extract document first.");
      return;
    }

    try {
      console.log("[ask-ai] question:", aiQuestion);
      const resp = await fetch(`${BACKEND_URL}/ask-ai`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: aiQuestion,
          full_cleaned_text: result.full_cleaned_text
        })
      });

      const text = await resp.text();
      let data;
      try { data = JSON.parse(text); } catch { data = { answer: text }; }

      console.log("[ask-ai] resp", resp.status, data);

      if (!resp.ok) {
        setError(data.error || `AI failed: ${resp.status}`);
        return;
      }

      setAiAnswer(data.answer ?? String(data));
    } catch (e) {
      console.error("[ask-ai] error", e);
      setError("AI request failed: " + e.message);
    }
  };

  // ---------- Anomaly / Risk ----------
  const handleDetectAnomalies = async () => {
    setAnomalyResult(null);
    setError("");
    if (!result) {
      setError("Extract a document first.");
      return;
    }
    try {
      const resp = await fetch(`${BACKEND_URL}/detect-anomalies`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_text: result.full_cleaned_text || result.full_raw_text || "" }),
      });
      const data = await resp.json();
      if (data.error) {
        setError(data.error);
        return;
      }
      setAnomalyResult(data);
    } catch (err) {
      setError("Anomaly detection failed: " + err.message);
    }
  };

  // Render progress bar
  const renderProgress = () => {
    const total = progress.total_to_process || progress.total || 0;
    const done = progress.processed || 0;
    const pct = total ? Math.round((done / total) * 100) : processing ? 10 : 0;
    return (
      <div style={{ marginTop: 18 }}>
        <div style={{ height: 18, background: "#e6e6e6", borderRadius: 12, overflow: "hidden" }}>
          <div style={{ width: `${pct}%`, height: "100%", background: "#9f7aea", transition: "width 300ms" }} />
        </div>
        <div style={{ textAlign: "center", marginTop: 10, color: "#4a5568" }}>
          {processing ? `Processing PDF... (${done}/${total || "?"}) — ${statusText}` : statusText}
        </div>
      </div>
    );
  };

  // Helper to display a short meta line
  const metaLine = (label, value) => (
    <div style={{ marginBottom: 6 }}>
      <strong>{label}</strong> {value ?? "(n/a)"}
    </div>
  );

  return (
    <div className="App">
      <div className="container">
        <h1>⚖️ Legal Document Intelligence</h1>
        <p className="subtitle">Upload a contract and interact with AI</p>

        <div className="upload-section" aria-live="polite">
          <input
            aria-label="Upload PDF"
            type="file"
            accept="application/pdf"
            onChange={handleFileChange}
          />
          <button
            className="btn-primary"
            disabled={uploading || processing || !file}
            onClick={handleUpload}
            aria-disabled={uploading || processing || !file}
          >
            {uploading ? "Uploading..." : processing ? "Processing..." : "📤 Extract Document"}
          </button>

          <button
            className="btn-secondary"
            disabled={!processing}
            onClick={cancelJob}
            style={{ marginLeft: 12 }}
          >
            Cancel
          </button>
        </div>

        {error && <div className="error-box" role="alert" style={{ marginTop: 16 }}>{error}</div>}

        {processing && renderProgress()}

        {/* RESULT / PREVIEW */}
        {!processing && result && (
          <div style={{ marginTop: 20 }}>
            <div className="info-box">
              {metaLine("File:", result.filename)}
              {metaLine("Pages:", result.pages)}
              {result.statistics && metaLine("OCR Used On:", (result.statistics.ocr_pages || []).join(", ") || "None")}
            </div>

            {/* Search */}
            <div className="search-box" style={{ marginTop: 10 }}>
              <h3>🔎 Search Document</h3>
              <form onSubmit={handleSearch} style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 8 }}>
                <input
                  className="search-input"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Type keywords or phrases..."
                  aria-label="Search query"
                />
                <button type="submit" className="btn-secondary" disabled={!searchQuery}>Search</button>
              </form>

              <div style={{ marginTop: 12 }}>
                {searchResults.length === 0 && <div style={{ color: "#6b7280" }}>No results yet.</div>}
                {searchResults.map((r, idx) => (
                  <div key={idx} className="result-item">
                    {r.line_number ? <small>Line {r.line_number}</small> : null}
                    <div style={{ marginTop: 6 }}>{r.text ?? r}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* AI Q&A */}
            <div className="ai-box" style={{ marginTop: 18 }}>
              <h3>💬 GenAI Legal Assistant</h3>
              <form onSubmit={handleAskAI} style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 8 }}>
                <input
                  className="ai-input"
                  placeholder="Ask legal questions based on this document..."
                  value={aiQuestion}
                  onChange={(e) => setAiQuestion(e.target.value)}
                  aria-label="AI question"
                />
                <button type="submit" className="btn-ai" disabled={!aiQuestion}>Ask AI</button>
              </form>

              <div style={{ marginTop: 12 }}>
                {aiAnswer ? (
                  <div className="ai-answer">{aiAnswer}</div>
                ) : (
                  <div style={{ color: "#6b7280" }}>No AI answer yet.</div>
                )}
              </div>
            </div>

            {/* Anomaly / Risk */}
            <div className="anomaly-box" style={{ marginTop: 18 }}>
              <h3>⚠️ Anomaly / Risk Detection</h3>
              <div style={{ marginTop: 8 }}>
                <button className="btn-warning" onClick={handleDetectAnomalies}>Detect Risks</button>
              </div>

              <div style={{ marginTop: 12 }}>
                {anomalyResult ? (
                  <>
                    <div style={{ marginBottom: 8 }}><strong>AI Feedback:</strong></div>
                    <div style={{ background: "#fff", padding: 12, borderRadius: 8 }}>{anomalyResult.ai_feedback || anomalyResult}</div>
                    <div style={{ marginTop: 12 }}>
                      <strong>Found Clauses</strong>
                      {Array.isArray(anomalyResult.found_clauses) && anomalyResult.found_clauses.length > 0 ? (
                        anomalyResult.found_clauses.map((c, i) => (
                          <div key={i} className="anomaly-item">
                            <div><strong>{c.pattern}</strong></div>
                            <div style={{ marginTop: 6 }}>{c.snippet}</div>
                          </div>
                        ))
                      ) : (
                        <div style={{ color: "#6b7280" }}>No risky clauses returned.</div>
                      )}
                    </div>
                  </>
                ) : (
                  <div style={{ color: "#6b7280" }}>No anomaly analysis yet.</div>
                )}
              </div>
            </div>

            {/* Sections / Summaries / Entities */}
            <div style={{ marginTop: 18 }}>
              <div className="entities-box">
                <h4>🔖 Sections</h4>
                {Array.isArray(result.sections) && result.sections.length > 0 ? (
                  result.sections.map((s, i) => (
                    <div key={i} style={{ marginBottom: 10, background: "#fff", padding: 10, borderRadius: 8 }}>
                      <strong>{s.title}</strong>
                      <div style={{ marginTop: 6, color: "#4a5568" }}>{s.snippet}</div>
                    </div>
                  ))
                ) : (
                  <div style={{ color: "#6b7280" }}>No sections extracted.</div>
                )}

                <h4 style={{ marginTop: 12 }}>📝 Summaries</h4>
                {Array.isArray(result.summaries) && result.summaries.length > 0 ? (
                  result.summaries.map((s, i) => (
                    <div key={i} className="result-item">{s}</div>
                  ))
                ) : (
                  <div style={{ color: "#6b7280" }}>No summaries available.</div>
                )}

                <h4 style={{ marginTop: 12 }}>📚 Entities</h4>
                <div className="entity-list" style={{ marginTop: 8 }}>
                  {Array.isArray(result.entities) && result.entities.length > 0 ? (
                    result.entities.slice(0, 50).map((e, i) => (
                      <div key={i} className="entity-tag">{e.text} <small style={{ marginLeft: 6, color: "#6b7280" }}>[{e.label}]</small></div>
                    ))
                  ) : (
                    <div style={{ color: "#6b7280" }}>No entities found.</div>
                  )}
                </div>
              </div>
            </div>

            {/* Preview text and pages */}
            <div style={{ marginTop: 18 }}>
              <h4>Preview (first 3 processed pages)</h4>
              {Array.isArray(result.pages_data) && result.pages_data.slice(0, 3).map((p, idx) => (
                <div key={idx} style={{ background: "#fff", padding: 12, marginBottom: 8, borderRadius: 6, border: "1px solid #eee" }}>
                  <div style={{ fontSize: 13, marginBottom: 6 }}>
                    <strong>Page {p.page_number}</strong> • method: {p.method} • chars: {p.char_count}
                  </div>
                  <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{p.text || "(no text captured)"}</pre>
                </div>
              ))}
              {!Array.isArray(result.pages_data) && <div style={{ color: "#666" }}>No page previews available.</div>}
            </div>

            <div style={{ marginTop: 16 }}>
              <h4>Full Cleaned Text</h4>
              <pre style={{ maxHeight: 360, overflow: "auto", whiteSpace: "pre-wrap", background: "#fafafa", padding: 12, borderRadius: 6 }}>
                {result.full_cleaned_text || "(empty)"}
              </pre>
            </div>
          </div>
        )}

        {/* No result yet */}
        {!processing && !result && (
          <div style={{ marginTop: 24, color: "#718096" }}>
            Choose a PDF and click <strong>Extract Document</strong>. The UI will upload, poll server status and show extracted text + tools here.
          </div>
        )}
      </div>
    </div>
  );
}
