# ⚖️ Legal Document Intelligence Platform

An **end‑to‑end AI-powered system** to upload legal documents (PDFs), extract and clean text, identify legal sections, perform advanced keyword search, enable **grounded GenAI Q&A**, and detect **risky / anomalous clauses**.

This project is **engineered-first**, not chatbot-first. AI is used **only after** deterministic processing to ensure reliability, explainability, and reduced hallucinations.

---

## 🚀 Why This Project Matters

Legal documents are long, noisy (OCR), and highly structured. Generic chatbots struggle with:

* Hallucinations
* Lack of clause references
* No explainability
* No enterprise workflow

This platform solves that by combining:

* OCR-aware text cleaning
* Regex-based legal structure extraction
* Contextual keyword search
* Responsible RAG-style GenAI usage
* Risk / anomaly detection

---

## 🧠 High-Level Architecture

```
PDF Upload (React)
   ↓
Flask Backend APIs
   ↓
OCR / Text Extraction
   ↓
Text Cleaning & Normalization
   ↓
Legal Section / Clause Extraction
   ↓
Keyword & Structured Search
   ↓
GenAI Q&A (Grounded)
   ↓
Risk / Anomaly Detection
```

---

## ✨ Key Features

### 📄 Document Ingestion

* Upload PDF contracts
* OCR support for scanned pages
* Background processing with job IDs
* Progress tracking via polling

### 🧹 OCR Text Cleaning

* Removes OCR noise (extra spaces, broken lines)
* Fixes common OCR errors (e.g. `|` → `I`)
* Removes page markers
* Produces clean, NLP-ready text

### 🧩 Legal Section Extraction

* Regex-based detection of:

  * Articles (Roman / numeric)
  * Sections (1, 2.3, etc.)
  * Clauses
  * Numbered headings
* Extracts:

  * Section number
  * Title
  * Content preview
  * Line number

### 🔎 Advanced Keyword Search

* Context-aware search (±60 chars)
* Case-insensitive, regex-safe
* De-duplicated snippets
* Section-aware results (where applicable)
* Occurrence counting

### 💬 GenAI Legal Assistant

* Ask questions **only after** extraction
* Uses cleaned text as grounding context
* Deterministic behavior for system prompts
* Designed to reduce hallucinations

### ⚠️ Risk / Anomaly Detection

* Detects potentially risky clauses

  * Penalty
  * Termination
  * Indemnity (extensible)
* Returns:

  * AI feedback
  * Clause snippets

### 📊 Transparent Outputs

* Page-wise previews
* Full cleaned document text
* Extracted entities
* Basic summaries & metadata

---

## 🖥️ Frontend (React)

* Single-page application
* State-driven UI (React Hooks)
* Async job orchestration (polling)
* Environment-based backend config
* Features:

  * Upload & progress tracking
  * Search
  * GenAI Q&A
  * Risk detection
  * Section & entity display

---

## ⚙️ Backend (Flask / Python)

### Core Modules

* `text_cleaner.py` – OCR text normalization
* `section_extractor.py` – Legal clause extraction
* `search.py` – Keyword & advanced search
* `summarizer.py` – Deterministic summaries
* `anomaly_detection.py` – Risk analysis

### Design Principles

* Separation of concerns
* Defensive input handling
* Regex before AI
* Explainable outputs
* Testable pipeline

---

## 🧪 Testing & Validation

* Standalone test harness for:

  * OCR cleaning
  * Section extraction
* Sample OCR text input
* Debug-friendly logging
* Optional output persistence

---

## 🔐 Security & Reliability

* API keys via environment variables
* No hardcoded secrets
* Regex-escaped user queries
* Deterministic AI temperature for system calls

---

## 🧰 Tech Stack

**Frontend**

* React
* HTML / CSS
* Fetch API

**Backend**

* Python
* Flask
* Regex / NLP preprocessing

**AI / GenAI**

* Groq API (LLM)
* Prompt grounding with document context

**Others**

* OCR tools
* Docker (optional)
* Git & GitHub

---

## ▶️ How to Run Locally

### Backend

```bash
cd backend
python app.py
```

### Frontend

```bash
cd webapp
npm install
npm start
```

Set environment variables:

```bash
export GROQ_API_KEY=your_key
export GROQ_MODEL=llama-3.1-8b-instant
```

---

