# app.py (RAG-enabled, improved) - cleaned & debugged (fixed)
import os
import io
import re
import uuid
import time
import pickle
import logging
import shutil
import subprocess
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv, dotenv_values
from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageEnhance

# added for safe filenames
from werkzeug.utils import secure_filename

# ---------------------- GUARANTEED .env LOADER (BOM-safe) ----------------------
BASE_DIR = os.path.dirname(__file__)
DOTENV_PATH = os.path.join(BASE_DIR, ".env")

print(">>> app.py starting")
print(">>> BASE_DIR:", BASE_DIR)
print(">>> DOTENV_PATH:", DOTENV_PATH)

p = Path(DOTENV_PATH)
if p.exists():
    try:
        raw = p.read_bytes()
        print(">>> .env exists, size:", p.stat().st_size)
        print(">>> .env first bytes:", raw[:4])
    except Exception as e:
        print(">>> Failed reading .env raw bytes:", e)
else:
    print(">>> .env NOT FOUND at expected location")

# Standard load
load_dotenv(dotenv_path=DOTENV_PATH, override=True)

# Robust parse + injection â€” fixes BOM or invisible prefixes in keys
try:
    parsed = dotenv_values(DOTENV_PATH) if p.exists() else {}
    fixed_keys = []
    for k, v in (parsed.items() if parsed else []):
        if k is None:
            continue
        clean_k = k.lstrip("\ufeff\xfe\xff").strip()
        if not clean_k:
            continue
        fixed_keys.append(clean_k)
        if v is None:
            continue
        if os.getenv(clean_k) is None or os.getenv(clean_k) == "":
            os.environ[clean_k] = v
    if fixed_keys:
        print(">>> Parsed .env keys (fixed):", fixed_keys)
    else:
        if p.exists():
            print(">>> dotenv_values returned empty/malformed dict (check .env content)")
except Exception as e:
    print(">>> dotenv parse error:", e)

_key = os.getenv("GROQ_API_KEY")
print(">>> GROQ_API_KEY present:", bool(_key))
if _key:
    print(">>> GROQ_API_KEY present (masked)")

# ---------------------- optional heavy libs (lazy) ----------------------
try:
    import numpy as np
except Exception:
    np = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("legal-ai-backend")

# ---------------- Config ----------------
class Config:
    # FIXED: add MAX_FILE_SIZE (was missing) â€” default 50MB
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 50 * 1024 * 1024))

    UPLOAD_FOLDER = "uploads"
    TESSERACT_PATH = os.getenv("TESSERACT_PATH", r"C:\Users\meghn\Downloads\tesseract.exe")
    TESSERACT_ENV = os.getenv("TESSERACT_CMD")
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "3"))
    OCR_DPI = int(os.getenv("OCR_DPI", "150"))
    OCR_THRESHOLD = int(os.getenv("OCR_THRESHOLD", "30"))
    MAX_PAGES_FULL = int(os.getenv("MAX_PAGES_FULL", "50"))

    CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
    CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
    TOP_K = int(os.getenv("RAG_TOP_K", "4"))

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# ---------------- Jobs ----------------
JOBS = {}
JOBS_LOCK = Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)

def get_job(job_id):
    with JOBS_LOCK:
        return JOBS.get(job_id)

def update_job(job_id, updates):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(updates)

# ---------------- Optional local utils fallback ----------------
try:
    from utils.text_cleaner import clean_ocr_text as clean_text_fn
    from utils.section_extractor import extract_legal_sections
    from utils.summarizer import summarize_entire_document
    from search_engine import keyword_search
    logger.info("âœ… Loaded local utils")
except Exception:
    logger.info("âš ï¸ Local utils not found, using internal fallbacks")

    def clean_text_fn(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\r\n?', '\n', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        lines = [ln.strip() for ln in text.split('\n')]
        return '\n'.join([ln for ln in lines if ln])

    def extract_legal_sections(text: str):
        out = []
        lines = text.splitlines()
        for i, ln in enumerate(lines):
            if not ln.strip():
                continue
            if ln.strip().isupper() and len(ln.strip()) < 80:
                snippet = "\n".join(lines[i:i+6])
                out.append({"title": ln.strip(), "snippet": snippet})
            elif re.match(r'^\s*(Section|Clause|Article)\b', ln, re.I):
                snippet = "\n".join(lines[i:i+6])
                out.append({"title": ln.strip(), "snippet": snippet})
        return out

    def summarize_entire_document(text: str):
        paras = [p.strip() for p in text.split('\n\n') if p.strip()]
        return paras[:3]

    def keyword_search(text: str, q: str):
        q = q.lower().strip()
        if not q:
            return []
        results = []
        for i, line in enumerate(text.split("\n")):
            if q in line.lower():
                results.append({"line_number": i + 1, "text": line.strip()})
                if len(results) >= 20:
                    break
        return results

# ---------------- spaCy optional ----------------
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    logger.info("âœ… spaCy loaded")
except Exception:
    nlp = None
    logger.info("âš ï¸ spaCy not available")

# ---------------- tesseract detection (improved) ----------------
def detect_tesseract(explicit_path=None):
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    env_path = Config.TESSERACT_ENV
    if env_path:
        candidates.append(env_path)
    which_path = shutil.which("tesseract")
    if which_path:
        candidates.append(which_path)

    # dedupe while preserving order
    seen = set()
    candidates = [c for c in candidates if c and not (c in seen or seen.add(c))]

    for p in candidates:
        try:
            # try to run the candidate; some candidates might be full paths or just commands
            out = subprocess.run([p, "-v"], capture_output=True, text=True, check=True, timeout=5)
            first_line = out.stdout.splitlines()[0] if out.stdout else (out.stderr.splitlines()[0] if out.stderr else p)
            logger.info("âœ… Tesseract found (candidate): %s", first_line)
            return p
        except Exception:
            logger.debug("Tesseract candidate failed: %s", p)

    # try generic 'tesseract' on PATH as last resort
    try:
        out = subprocess.run(["tesseract", "-v"], capture_output=True, text=True, check=True, timeout=5)
        logger.info("âœ… Tesseract available via PATH: %s", out.stdout.splitlines()[0] if out.stdout else "tesseract")
        return shutil.which("tesseract") or "tesseract"
    except Exception:
        logger.warning("Tesseract not detected on system PATH")
    return None

_detected_tesseract = detect_tesseract(Config.TESSERACT_PATH)
tesseract_available = bool(_detected_tesseract)
if tesseract_available:
    try:
        pytesseract.pytesseract.tesseract_cmd = _detected_tesseract
    except Exception:
        logger.exception("Failed to set pytesseract cmd, but binary exists")
logger.info(f"Tesseract available: {tesseract_available}")

# ---------------- Helpers: OCR + extraction ----------------
def is_text_rich(text, threshold=Config.OCR_THRESHOLD):
    if not text:
        return False
    text_chars = len(re.sub(r'\s+', '', text))
    return text_chars >= threshold

def ocr_image(pil_img):
    if not tesseract_available:
        return ""
    try:
        if max(pil_img.size) > 2500:
            ratio = 2500 / max(pil_img.size)
            pil_img = pil_img.resize((int(pil_img.size[0]*ratio), int(pil_img.size[1]*ratio)), Image.Resampling.LANCZOS)
        pil_img = pil_img.convert('L')
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.5)
        config = "--psm 6 --oem 3"
        return pytesseract.image_to_string(pil_img, lang="eng", config=config)
    except Exception:
        logger.exception("OCR error")
        return ""

def extract_text_native(page):
    try:
        text = page.get_text("text") or ""
        if is_text_rich(text):
            return text, "native_text"
        blocks = page.get_text("blocks")
        if blocks:
            blocks_sorted = sorted([b for b in blocks if b[4].strip()], key=lambda b: (b[1], b[0]))
            text = "\n".join([b[4].strip() for b in blocks_sorted])
            if is_text_rich(text):
                return text, "blocks"
        return "", "empty"
    except Exception:
        logger.exception("Native extraction error")
        return "", "error"

def extract_text_ocr(page):
    if not tesseract_available:
        return "", "ocr_unavailable"
    try:
        pix = page.get_pixmap(dpi=Config.OCR_DPI)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = ocr_image(img)
        if is_text_rich(text):
            return text, "ocr"
        return "", "ocr_empty"
    except Exception:
        logger.exception("OCR extraction error")
        return "", "ocr_error"

def process_page(doc, page_num):
    try:
        page = doc.load_page(page_num)
        text, method = extract_text_native(page)
        if not is_text_rich(text) and tesseract_available:
            text, method = extract_text_ocr(page)
        return {"page_number": page_num+1, "text": text or "", "method": method, "char_count": len(text or "")}
    except Exception:
        logger.exception(f"Error processing page {page_num+1}")
        return {"page_number": page_num+1, "text": "", "method": "error", "char_count": 0}

# ---------------- RAG helpers (chunking + embeddings + persist) ----------------
def chunk_text(text, chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP):
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = splitter.split_text(text)
        return [d.page_content if hasattr(d, "page_content") else d for d in docs]
    except Exception:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            j = min(i + chunk_size, len(words))
            chunk = " ".join(words[i:j])
            chunks.append(chunk)
            i = j - chunk_overlap if (j - chunk_overlap) > i else j
        return chunks

def create_embeddings(texts):
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectors = embedder.embed_documents(texts)
        import numpy as _np
        vectors = _np.array(vectors, dtype=_np.float32)
        return vectors, "huggingface_langchain"
    except Exception:
        logger.info("HuggingFace LangChain embedder not available")

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        if np is not None:
            return np.array(vectors, dtype=np.float32), "sentence_transformers"
        else:
            import numpy as _np
            return _np.array(vectors, dtype=_np.float32), "sentence_transformers"
    except Exception:
        logger.info("sentence-transformers not available")

    return None, None

def persist_vectors(job_id, chunks, vectors):
    try:
        import numpy as _np
        vec_path = os.path.join(Config.UPLOAD_FOLDER, f"{job_id}_vectors.npz")
        chunk_path = os.path.join(Config.UPLOAD_FOLDER, f"{job_id}_chunks.pkl")
        _np.savez_compressed(vec_path, vectors=vectors)
        with open(chunk_path, "wb") as fh:
            pickle.dump(chunks, fh)
        return vec_path, chunk_path
    except Exception:
        logger.exception("Failed to persist vectors")
        return None, None

def load_vectors(job_id):
    vec_path = os.path.join(Config.UPLOAD_FOLDER, f"{job_id}_vectors.npz")
    chunk_path = os.path.join(Config.UPLOAD_FOLDER, f"{job_id}_chunks.pkl")
    if not os.path.exists(vec_path) or not os.path.exists(chunk_path):
        return None, None
    try:
        import numpy as _np
        data = _np.load(vec_path)
        vectors = data["vectors"]
        with open(chunk_path, "rb") as fh:
            chunks = pickle.load(fh)
        return chunks, vectors
    except Exception:
        logger.exception("Failed to load vectors")
        return None, None

def similarity_search(chunks, vectors, query, top_k=Config.TOP_K):
    if vectors is None or (np is None):
        return []
    q_vec, name = create_embeddings([query])
    if q_vec is None:
        return []
    q_vec = q_vec[0] if len(q_vec.shape) > 1 else q_vec
    norms = np.linalg.norm(vectors, axis=1) * (np.linalg.norm(q_vec) + 1e-12)
    sims = (vectors @ q_vec) / (norms + 1e-12)
    top_idx = np.argsort(-sims)[:top_k]
    results = [{"text": chunks[int(i)], "score": float(sims[int(i)])} for i in top_idx]
    return results

# ---------------- PDF processing job ----------------
def process_pdf_job(job_id, file_path):
    try:
        update_job(job_id, {"status": "processing", "progress": 0})
        logger.info(f"[JOB {job_id}] Processing: {file_path}")

        doc = fitz.open(file_path)
        total_pages = len(doc)
        update_job(job_id, {"total_pages": total_pages})

        if total_pages <= Config.MAX_PAGES_FULL:
            pages_to_process = list(range(total_pages))
            strategy = "full"
        else:
            pages_to_process = list(range(10)) + list(range(total_pages-10, total_pages))
            strategy = f"sampled_{len(pages_to_process)}_of_{total_pages}"
        update_job(job_id, {"strategy": strategy, "pages_to_process": len(pages_to_process)})

        pages_data = []
        methods_used = {}
        ocr_pages = []

        for idx, page_num in enumerate(pages_to_process):
            res = process_page(doc, page_num)
            pages_data.append(res)
            method = res.get("method", "unknown")
            methods_used[method] = methods_used.get(method, 0) + 1
            if "ocr" in method:
                ocr_pages.append(res["page_number"])
            progress_pct = int(((idx+1)/len(pages_to_process))*100)
            update_job(job_id, {"progress": progress_pct, "processed_pages": idx+1})
            if (idx+1) % 5 == 0:
                logger.info(f"[JOB {job_id}] {idx+1}/{len(pages_to_process)} pages done")

        pages_data.sort(key=lambda x: x["page_number"])
        raw_text = "\n\n".join([f"[Page {p['page_number']}]\n{p['text']}" for p in pages_data if p.get("text")])
        cleaned = clean_text_fn(raw_text)

        sections = extract_legal_sections(cleaned) or []
        summaries = summarize_entire_document(cleaned) or []

        entities = []
        if nlp:
            try:
                doc_nlp = nlp(cleaned[:150000])
                entities = [{"text": ent.text, "label": ent.label_} for ent in doc_nlp.ents]
            except Exception:
                logger.exception("spaCy extraction failed")

        statistics = {"methods_used": methods_used, "ocr_pages_count": len(ocr_pages), "ocr_pages": ocr_pages, "total_characters": sum(p.get("char_count", 0) for p in pages_data)}

        # RAG setup: chunk + embeddings + persist
        chunks = chunk_text(cleaned)
        vectors, embedder_name = create_embeddings(chunks)
        vec_paths = (None, None)
        if vectors is not None:
            vec_paths = persist_vectors(job_id, chunks, vectors)
            logger.info(f"[JOB {job_id}] persisted vectors: {vec_paths}")
        else:
            logger.info(f"[JOB {job_id}] embeddings not available; RAG disabled for this job")

        result = {
            "success": True,
            "filename": os.path.basename(file_path),
            "pages": total_pages,
            "pages_processed": len(pages_to_process),
            "processing_strategy": strategy,
            "pages_data": pages_data,
            "full_raw_text": raw_text,
            "full_cleaned_text": cleaned,
            "sections": sections,
            "summaries": summaries,
            "entities": entities,
            "statistics": statistics,
            "rag": {"chunks_count": len(chunks), "embedder": embedder_name, "vectors_saved": bool(vectors is not None)}
        }

        update_job(job_id, {"status": "completed", "progress": 100, "result": result})
        doc.close()
        logger.info(f"[JOB {job_id}] Completed")
    except Exception as e:
        logger.exception(f"[JOB {job_id}] processing error")
        update_job(job_id, {"status": "error", "error": str(e)})

# ---------------- Create Flask app and routes ----------------
app = Flask(__name__)
# allow local dev origins commonly used
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    supports_credentials=False
)

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Hello from Flask (backend)"}), 200

@app.route("/api/ping", methods=["GET"])
def ping():
    return jsonify({"ping": "pong"}), 200

@app.route("/api/echo", methods=["POST"])
def echo():
    data = request.get_json(silent=True) or {}
    return jsonify({"you_sent": data}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok"), 200

@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    status = job.get("status")
    if status == "completed":
        return jsonify({"status": "completed", "result": job.get("result")})
    if status == "error":
        return jsonify({"status": "error", "error": job.get("error", "Unknown error")})
    return jsonify({"status": status, "progress": job.get("progress", 0), "total_pages": job.get("total_pages"), "processed_pages": job.get("processed_pages"), "pages_to_process": job.get("pages_to_process")})

@app.route("/extract", methods=["POST"])
def extract():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    # defend against empty filename
    if not (file and getattr(file, "filename", None)):
        return jsonify({"error": "Invalid file upload"}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files allowed"}), 400

    # Robust file size check: handle FileStorage.stream or file-like objects
    try:
        # prefer underlying stream seek/tell
        stream = getattr(file, "stream", file)
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(0)
    except Exception:
        try:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(0)
        except Exception:
            # fallback: unknown size - allow (but could be risky)
            size = 0

    # FIXED: use correct Config.MAX_FILE_SIZE (typo before used MAX_FILE_size)
    if size and size > Config.MAX_FILE_SIZE:
        return jsonify({"error": "File too large"}), 400

    try:
        job_id = uuid.uuid4().hex
        # protect filename
        safe_name = secure_filename(file.filename)
        filename = f"{job_id}_{safe_name}"
        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(file_path)
        with JOBS_LOCK:
            JOBS[job_id] = {"job_id": job_id, "status": "queued", "filename": file.filename, "file_path": file_path, "progress": 0, "created_at": time.time()}
        EXECUTOR.submit(process_pdf_job, job_id, file_path)
        logger.info(f"Job created {job_id}")
        return jsonify({"job_id": job_id, "status": "queued"}), 202
    except Exception as e:
        # improved logging message
        logger.exception("upload error: %s", str(e))
        return jsonify({"error": "Upload failed", "detail": str(e)}), 500

@app.route("/search", methods=["POST", "GET"])
def search():
    try:
        if request.method == "GET":
            q = request.args.get("q") or request.args.get("query") or ""
            text = request.args.get("text") or request.args.get("full_cleaned_text") or ""
            if not q or not text:
                return jsonify({"error": "Provide ?q=...&text=..."}), 400
            results = keyword_search(text, q)
            return jsonify({"results": results}), 200

        data = {}
        if request.is_json:
            data = request.get_json() or {}
        else:
            data = request.form.to_dict() or {}
            if not data and request.data:
                try:
                    import json
                    data = json.loads(request.data.decode("utf-8") or "{}")
                except Exception:
                    data = {}

        query = (data.get("query") or data.get("q") or "").strip()
        text = (data.get("full_cleaned_text") or data.get("text") or data.get("document_text") or "").strip()
        if not query or not text:
            return jsonify({"error": "Missing query or text"}), 400

        results = keyword_search(text, query)
        normalized = []
        for r in results:
            if isinstance(r, str):
                normalized.append({"text": r})
            elif isinstance(r, dict):
                normalized.append({"text": r.get("text", ""), "line_number": r.get("line_number")})
            else:
                normalized.append({"text": str(r)})
        return jsonify({"results": normalized}), 200
    except Exception:
        logger.exception("search error")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/ask-ai", methods=["POST"])
def ask_ai():
    try:
        data = {}
        if request.is_json:
            data = request.get_json() or {}
        else:
            data = request.form.to_dict() or {}
            if not data and request.data:
                try:
                    import json
                    data = json.loads(request.data.decode("utf-8") or "{}")
                except Exception:
                    data = {}

        question = (data.get("question") or data.get("q") or "").strip()
        if not question:
            return jsonify({"error": "Missing question"}), 400

        job_id = data.get("job_id") or data.get("job") or None
        context_text = data.get("full_cleaned_text") or data.get("full_raw_text") or ""
        pages = data.get("pages_data") or []

        rag_context = ""
        if job_id:
            chunks, vectors = load_vectors(job_id)
            if chunks is not None and vectors is not None:
                logger.info(f"[ask-ai] using stored vectors for job {job_id}")
                top = similarity_search(chunks, vectors, question, top_k=Config.TOP_K)
                rag_context = "\n\n".join([f"(score {t['score']:.3f})\n{t['text']}" for t in top])
        else:
            if context_text:
                chunks = chunk_text(context_text)
                vectors, _ = create_embeddings(chunks)
                if vectors is not None:
                    top = similarity_search(chunks, vectors, question, top_k=Config.TOP_K)
                    rag_context = "\n\n".join([f"(score {t['score']:.3f})\n{t['text']}" for t in top])

        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                from groq import Groq
                client = Groq(api_key=groq_key)

                final_context = rag_context if rag_context else (
                    "\n\n".join([p.get("text", "") for p in pages[:4]]) if pages else (context_text[:15000] if context_text else "")
                )
                prompt = f"""You are a legal assistant. Use ONLY the provided context to answer. If the document does not contain the answer, reply: "The document does not provide this information."

Context:
{final_context}

Question:
{question}

Answer concisely (one paragraph) and reference context lines if possible."""

                completion = client.chat.completions.create(
                    model=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
                    messages=[
                        {"role": "system", "content": "You are a legal assistant. Answer only using context."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=float(os.getenv("GROQ_TEMP", "0.0")),
                    max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "512")),
                )

                def extract_completion_text(c):
                    try:
                        if hasattr(c, "choices") and getattr(c, "choices"):
                            choice = c.choices[0]
                            msg = getattr(choice, "message", None)
                            if isinstance(msg, dict):
                                return msg.get("content") or msg.get("text") or str(c)
                            if msg is not None and hasattr(msg, "get"):
                                return msg.get("content") or msg.get("text") or str(c)
                            if msg is not None and hasattr(msg, "content"):
                                return msg.content
                            if hasattr(choice, "text"):
                                return choice.text
                        if hasattr(c, "content"):
                            return getattr(c, "content")
                        if isinstance(c, dict):
                            if "choices" in c and c["choices"]:
                                ch = c["choices"][0]
                                if isinstance(ch, dict):
                                    if "message" in ch and isinstance(ch["message"], dict):
                                        return ch["message"].get("content") or ch["message"].get("text") or str(c)
                                    if "text" in ch:
                                        return ch["text"]
                            return str(c)
                    except Exception:
                        pass
                    return str(c)

                answer = extract_completion_text(completion) or "No answer returned by LLM."
                answer = str(answer).strip()
                return jsonify({"answer": answer, "used_context": bool(final_context)}), 200

            except Exception as e:
                logger.exception("Groq call failed (detailed)")
                logger.warning("Groq error: %s", str(e)[:1000])

        if rag_context:
            summary_lines = ["**Extractive answer (RAG):**"]
            for i, chunk in enumerate(rag_context.split("\n\n")[:4]):
                summary_lines.append(f"- {chunk.strip()}")
            return jsonify({"answer": "\n".join(summary_lines), "used_context": True, "note": "LLM unavailable or fallback used"}), 200

        if context_text:
            q = question.lower()
            matches = []
            lines = context_text.splitlines()
            for i, line in enumerate(lines):
                tokens = [w for w in re.findall(r"[a-zA-Z0-9]{3,}", q)]
                if any(tok in line.lower() for tok in tokens):
                    start = max(0, i-2)
                    snippet = "\n".join(lines[start:start+5]).strip()
                    matches.append({"line_number": i+1, "text": snippet})
                    if len(matches) >= 6:
                        break
            if matches:
                ans = ["**Extractive answer (fallback):**"]
                for m in matches:
                    ans.append(f"- (line {m['line_number']}) {m['text']}")
                return jsonify({"answer": "\n".join(ans), "used_context": True, "note": "fallback extractive"}), 200

        return jsonify({"answer": "No matches found in document. Try a different wording.", "used_context": False}), 200

    except Exception:
        logger.exception("ask-ai unexpected error")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/detect-anomalies", methods=["POST"])
def detect_anomalies():
    data = {}
    if request.is_json:
        data = request.get_json() or {}
    else:
        data = request.form.to_dict() or {}
        if not data and request.data:
            try:
                import json
                data = json.loads(request.data.decode("utf-8") or "{}")
            except Exception:
                data = {}

    text = data.get("document_text") or data.get("full_cleaned_text") or data.get("full_raw_text") or ""
    if not text:
        return jsonify({"error": "document_text missing"}), 400
    if not os.getenv("GROQ_API_KEY"):
        patterns = ["termination", "breach", "penalty", "liability", "confidentiality", "non-compete", "arbitration", "indemnity", "governing law", "force majeure"]
        found = []
        lowered = text.lower()
        for ptn in patterns:
            idx = lowered.find(ptn)
            if idx != -1:
                snippet = text[max(0, idx-120):idx+200]
                found.append({"pattern": ptn, "snippet": snippet})
        if not found:
            return jsonify({"found_clauses": [], "ai_feedback": "No risky clauses found."})
        return jsonify({"found_clauses": found, "ai_feedback": "Pattern-based detection returned. Set GROQ_API_KEY for AI summarization."})
    try:
        from langchain_groq import ChatGroq
        model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        llm = ChatGroq(model=model_name, temperature=0, max_tokens=512)
        patterns = ["termination", "breach", "penalty", "liability", "confidentiality", "non-compete", "arbitration", "indemnity", "governing law", "force majeure"]
        found = []
        lowered = text.lower()
        for ptn in patterns:
            idx = lowered.find(ptn)
            if idx != -1:
                snippet = text[max(0, idx-120):idx+200]
                found.append({"pattern": ptn, "snippet": snippet})
        if not found:
            return jsonify({"found_clauses": [], "ai_feedback": "No risky clauses found"})
        summary = "\n\n".join([f"{i+1}. {c['pattern']}:\n{c['snippet']}" for i, c in enumerate(found[:5])])
        prompt = f"Analyze these contract clauses for legal risks:\n\n{summary}\n\nProvide a concise risk summary:"
        response = llm.invoke(prompt)
        ai_feedback = response.content if hasattr(response, "content") else str(response)
        return jsonify({"found_clauses": found, "ai_feedback": ai_feedback})
    except Exception:
        logger.exception("anomaly LLM failed")
        return jsonify({"found_clauses": [], "ai_feedback": "Automatic summarization failed; only clauses returned."})

# ---------------- Debug: print Flask url map ----------------
print(">>> Flask URL map:")
try:
    print(app.url_map)
except Exception as e:
    print(">>> Failed to print url_map:", e)

# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    logger.info("Starting Legal Document Intelligence Platform (RAG enabled)")
    logger.info(f"Tesseract: {tesseract_available} (cmd={pytesseract.pytesseract.tesseract_cmd if tesseract_available else None})")
    logger.info(f"Max workers: {Config.MAX_WORKERS}")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), debug=False)

