# rag_utils.py
import os
import json
import uuid
import time
import shutil
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dotenv import load_dotenv
from azure_client import get_client

# ----- Load .env from project folder -----
here = Path(__file__).resolve().parent
load_dotenv(dotenv_path=str(here / ".env"), override=True)

# ----- Constants -----
EMBED_DEPLOYMENT = os.getenv("AZURE_EMBED_DEPLOYMENT", "text-embedding-3-large")
STORE_DIR = here / "data" / "vector_store"            # persistent store
DOCS_DIR = here / "data" / "vector_store" / "docs"    # we keep original copies
STORE_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# ======= Utilities =======

def _hash_path(path: Path) -> str:
    h = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    return h[:12]

def _norm_text(s: str) -> str:
    return " ".join((s or "").split())

def _read_text_from_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            return "[RAG error] Missing PyPDF2. Run: pip install pypdf2"
        out = []
        reader = PdfReader(str(path))
        for p in reader.pages:
            out.append(p.extract_text() or "")
        return "\n".join(out)
    if ext == ".docx":
        try:
            import docx
        except ImportError:
            return "[RAG error] Missing python-docx. Run: pip install python-docx"
        d = docx.Document(str(path))
        return "\n".join([p.text for p in d.paragraphs])
    return "[RAG error] Unsupported file type. Upload PDF/DOCX/TXT."

def _chunk(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += max(1, chunk_size - overlap)
    return [c for c in chunks if c.strip()]

def _embed(texts: List[str]) -> np.ndarray:
    client = get_client()
    resp = client.embeddings.create(
        model=EMBED_DEPLOYMENT,
        input=texts
    )
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype=np.float32)

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T

def _persist_index(doc_id: str, vectors: np.ndarray, chunks: List[str], meta: dict):
    # vectors + chunks
    np.savez(STORE_DIR / f"{doc_id}.npz",
             vectors=vectors,
             chunks=np.array(chunks, dtype=object))
    # metadata
    with open(STORE_DIR / f"{doc_id}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def _load_index(doc_id: str) -> Tuple[np.ndarray, List[str], dict]:
    data = np.load(STORE_DIR / f"{doc_id}.npz", allow_pickle=True)
    vectors = data["vectors"]
    chunks = list(data["chunks"])
    with open(STORE_DIR / f"{doc_id}.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return vectors, chunks, meta

def _doc_id_from_path(path: Path) -> str:
    stem = path.stem[:32]
    salt = _hash_path(path)
    return f"{stem}-{salt}"

def _copy_to_docs(src_path: Path) -> Path:
    # keep a copy by unique name to stabilize doc_id across sessions
    dest = DOCS_DIR / f"{_doc_id_from_path(src_path)}{src_path.suffix.lower()}"
    if not dest.exists():
        shutil.copy2(str(src_path), str(dest))
    return dest

def _list_indexed_docs() -> List[Dict[str, str]]:
    results = []
    for json_file in STORE_DIR.glob("*.json"):
        try:
            meta = json.loads(json_file.read_text(encoding="utf-8"))
            doc_id = json_file.stem
            title = meta.get("title") or doc_id
            results.append({"doc_id": doc_id, "title": title})
        except Exception:
            continue
    # stable sort by title
    return sorted(results, key=lambda x: x["title"].lower())

# ======= Public API used by app.py =======

def refresh_doc_list() -> List[str]:
    """
    Returns the dropdown choices: ["All documents", <doc_title 1>, <doc_title 2>, ...]
    """
    docs = _list_indexed_docs()
    choices = ["All documents"] + [d["title"] for d in docs]
    return choices

def ensure_index(files: Optional[List], progress=lambda x: None) -> str:
    """
    Build index for each uploaded file if not present.
    Returns an informational message with how many docs were processed.
    """
    if not files:
        return "No files to index (upload PDF/DOCX/TXT)."

    processed = 0
    for f in files:
        # Gradio File returns a TempFile object with .name
        src = Path(getattr(f, "name", f))
        dest = _copy_to_docs(src)
        text = _read_text_from_file(dest)
        if text.startswith("[RAG error]"):
            return text

        chunks = _chunk(_norm_text(text))
        if not chunks:
            return f"[RAG] No text extracted from: {dest.name}"

        doc_id = _doc_id_from_path(dest)
        npz_path = STORE_DIR / f"{doc_id}.npz"
        json_path = STORE_DIR / f"{doc_id}.json"

        # skip if already indexed & up-to-date (simple existence check)
        if npz_path.exists() and json_path.exists():
            processed += 1
            continue

        vectors = _embed(chunks)
        meta = {
            "doc_id": doc_id,
            "title": dest.name,
            "chunk_count": len(chunks),
            "created_at": int(time.time())
        }
        _persist_index(doc_id, vectors, chunks, meta)
        processed += 1

    return f"[RAG] Indexed/validated {processed} document(s)."

def _search_doc(doc_id: str, question: str, top_k: int = 4) -> str:
    """
    Search within a single document; returns context string.
    """
    vectors, chunks, meta = _load_index(doc_id)
    qvec = _embed([question])  # (1, d)
    sims = _cosine_sim(qvec, vectors).flatten()
    top_idx = sims.argsort()[-top_k:][::-1]
    context = "\n\n".join(chunks[i] for i in top_idx)
    title = meta.get("title", doc_id)
    header = f"### Source: {title}\n"
    return header + context

def _search_all(question: str, top_k_per_doc: int = 2) -> str:
    """
    Retrieve top contexts from all indexed docs and concatenate.
    """
    docs = _list_indexed_docs()
    if not docs:
        return ""
    contexts = []
    for d in docs:
        vectors, chunks, meta = _load_index(d["doc_id"])
        qvec = _embed([question])
        sims = _cosine_sim(qvec, vectors).flatten()
        top_idx = sims.argsort()[-top_k_per_doc:][::-1]
        ctx = "\n\n".join(chunks[i] for i in top_idx)
        title = meta.get("title", d["doc_id"])
        contexts.append(f"### Source: {title}\n{ctx}")
    return "\n\n---\n\n".join(contexts)

def answer_question(question: str, selected_title: str) -> str:
    """
    Build retrieval context (one doc or all), then ask the Chat model.
    """
    if not question or not question.strip():
        return "Type a question."

    docs = _list_indexed_docs()
    if not docs:
        return "No indexed documents yet. Upload and click 'Index Documents'."

    if selected_title == "All documents":
        context = _search_all(question)
        label = "multiple documents"
    else:
        # map title back to doc_id
        match = next((d for d in docs if d["title"] == selected_title), None)
        if not match:
            return "The selected document is not indexed anymore. Click 'Refresh list'."
        context = _search_doc(match["doc_id"], question)
        label = selected_title

    if not context.strip():
        return "I couldn’t retrieve any relevant context. Try re-indexing or ask a broader question."

    client = get_client()
    chat_dep = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o-mini")

    prompt = (
        "You are a helpful document QA assistant. "
        "Answer ONLY using the provided context. If the answer is not in the context, say: "
        "\"I don't see that in the document.\" Be concise.\n\n"
        f"Context from {label}:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    resp = client.chat.completions.create(
        model=chat_dep,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=500  # no temperature param (some models only support default=1)
    )
    return resp.choices[0].message.content
