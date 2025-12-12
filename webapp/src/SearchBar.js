// src/SearchBar.js
import React, { useState } from "react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:5000";

export default function SearchBar({ result, onResults, setError }) {
  const [q, setQ] = useState("");
  const [loading, setLoading] = useState(false);

  async function doSearch(e) {
    if (e && e.preventDefault) e.preventDefault(); // STOP full-page reload
    setError && setError("");
    onResults && onResults([]); // reset prior results

    if (!q.trim()) {
      setError && setError("Enter a search query");
      return;
    }
    if (!result || !(result.full_cleaned_text || result.full_raw_text)) {
      setError && setError("Extract a document first");
      return;
    }

    setLoading(true);
    try {
      const resp = await fetch(`${BACKEND_URL}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: q,
          // send both keys; backend accepts either
          full_cleaned_text: result.full_cleaned_text || result.full_raw_text || "",
          text: result.full_cleaned_text || result.full_raw_text || ""
        })
      });

      const text = await resp.text();
      // try parse JSON even when non-200 to show helpful error
      let data;
      try { data = JSON.parse(text); } catch { data = text; }

      if (!resp.ok) {
        const errMsg = (data && data.error) ? data.error : `Search failed: ${resp.status} ${resp.statusText}`;
        setError && setError(errMsg);
        return;
      }

      const results = (data && data.results) ? data.results : [];
      onResults && onResults(results);
    } catch (err) {
      setError && setError("Search request failed: " + err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ marginTop: 8 }}>
      <form onSubmit={doSearch} style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <input
          className="search-input"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Type keywords or phrases..."
          aria-label="Search query"
        />
        <button type="submit" className="btn-secondary" disabled={loading || !q.trim()}>
          {loading ? "Searching…" : "Search"}
        </button>
      </form>
    </div>
  );
}
