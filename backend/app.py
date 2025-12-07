from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# === Load ENV ===
load_dotenv()

# === AI + NLP ===
import spacy
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# === Utils Fallback ===
try:
    from utils.text_cleaner import clean_ocr_text
    from utils.section_extractor import extract_legal_sections
    from utils.summarizer import summarize_entire_document
    from search_engine import keyword_search
except:
    def clean_ocr_text(text): return text.strip()
    def extract_legal_sections(text): return []
    def summarize_entire_document(text): return []
    def keyword_search(text, q):
        q = q.lower()
        return [line for line in text.split("\n") if q in line.lower()][:10]

# === Flask ===
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === Config ===
# === Configuration ===

class Config:
    MAX_FILE_SIZE = 50 * 1024 * 1024
    ALLOWED_EXT = {"pdf"}
    TESSERACT_PATH = r"C:\Users\meghn\Downloads\tesseract.exe"
    UPLOAD_FOLDER = "uploads"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# === SpaCy ===
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None


# === Tesseract ===
if os.path.exists(Config.TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_PATH


# ============================================================
#                         HOME
# ============================================================
@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "groq_key_found": bool(os.getenv("GROQ_API_KEY"))
    })


# ============================================================
#                      PDF EXTRACTION
# ============================================================
@app.route("/extract", methods=["POST"])
def extract():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF allowed"}), 400

    # File size check
    file.seek(0, os.SEEK_END)
    if file.tell() > Config.MAX_FILE_SIZE:
        return jsonify({"error": "File size exceeds 50MB"}), 400
    file.seek(0)

    try:
        pdf_bytes = file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        raw_text = ""
        ocr_pages = []

        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text("text")

            if not text.strip():
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img)
                ocr_pages.append(i + 1)

            raw_text += f"\n\n[Page {i+1}]\n{text}"

        cleaned = clean_ocr_text(raw_text)
        sections = extract_legal_sections(cleaned)
        summaries = summarize_entire_document(cleaned)

        # Entity Extraction
        entities = []
        if nlp:
            doc_nlp = nlp(cleaned[:100000])
            entities = [{"text": e.text, "label": e.label_} for e in doc_nlp.ents]

        return jsonify({
            "filename": file.filename,
            "pages": len(doc),
            "ocr_pages": ocr_pages,
            "full_raw_text": raw_text,
            "full_cleaned_text": cleaned,
            "sections": sections[:10],
            "summaries": summaries,
            "entities": entities[:50]
        })

    except Exception as e:
        logger.exception("Extraction Error")
        return jsonify({"error": str(e)}), 500


# ============================================================
#                         SEARCH
# ============================================================
@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query")
    text = data.get("text")

    if not query or not text:
        return jsonify({"error": "query and text required"}), 400

    results = keyword_search(text, query)
    return jsonify({"results": results})


# ============================================================
#                     AI QUESTION ANSWER
# ============================================================
@app.route("/ask-ai", methods=["POST"])
def ask_ai():
    data = request.get_json()
    question = data.get("question", "").strip()
    doc_text = data.get("document_text", "").strip()

    if not question:
        return jsonify({"error": "Question required"}), 400
    if not doc_text:
        return jsonify({"error": "Document text missing"}), 400

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return jsonify({"error": "GROQ_API_KEY missing in .env"}), 500

    try:
        # Chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = splitter.create_documents([doc_text])

        # Embeddings
        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embedder)

        matches = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([m.page_content for m in matches])

        # Groq LLM (UPDATED MODEL)
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=1024)

        prompt = f"""
You are a legal expert. Answer concisely based on the document context.

Context:
{context}

Question: {question}

Answer:
"""

        # invoke() FIXED
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else response

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
#                 ANOMALY / RISK DETECTION
# ============================================================
@app.route("/detect-anomalies", methods=["POST"])
def detect_anomalies():
    data = request.get_json()
    text = data.get("document_text", "")

    if not text:
        return jsonify({"error": "document_text missing"}), 400

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return jsonify({"error": "GROQ_API_KEY missing"}), 500

    patterns = [
        "termination", "breach", "penalty", "liability",
        "confidentiality", "non-compete", "arbitration",
        "governing law", "force majeure", "indemnity"
    ]

    found = []
    lowered = text.lower()

    for p in patterns:
        idx = lowered.find(p)
        if idx != -1:
            snippet = text[max(0, idx - 120): idx + 200]
            found.append({"pattern": p, "snippet": snippet})

    if not found:
        return jsonify({
            "found_clauses": [],
            "ai_feedback": "No risky clauses found."
        })

    summary_text = "\n\n".join(
        [f"{i+1}. {c['pattern']}:\n{c['snippet']}" for i, c in enumerate(found[:5])]
    )

    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=1024)

        prompt = f"""
Analyze these contract clauses for legal risks:

{summary_text}

Provide a concise risk summary:
"""

        response = llm.invoke(prompt)
        ai_feedback = response.content if hasattr(response, "content") else response

        return jsonify({
            "found_clauses": found,
            "ai_feedback": ai_feedback
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
