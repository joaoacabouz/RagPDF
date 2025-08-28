# app.py
import os
import json
import faiss
import requests
import numpy as np
import streamlit as st
from io import BytesIO
from typing import List, Dict, Tuple
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ----------------------- Constants & Paths -----------------------
APP_TITLE = "Chat com Documentos (RAG) • Streamlit + FAISS + Sentence-Transformers + Ollama"
PDF_DIR = "pdfs"
VSTORE_DIR = "vectorstore"
INDEX_PATH = os.path.join(VSTORE_DIR, "index.faiss")
META_PATH = os.path.join(VSTORE_DIR, "meta.json")
DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = "gemma3:1b"  # default model

# ----------------------- Utilities -----------------------
def ensure_dirs():
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(VSTORE_DIR, exist_ok=True)

def ollama_is_up() -> bool:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def ollama_pull(model: str):
    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/pull", json={"name": model}, stream=True, timeout=None)
        last = ""
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
                status = data.get("status") or data.get("detail") or ""
                last = status
            except Exception:
                pass
        return True, last
    except Exception as e:
        return False, str(e)

def load_embedder(name: str = DEFAULT_EMBEDDER) -> SentenceTransformer:
    return SentenceTransformer(name)

def extract_text_from_pdf(filelike: BytesIO, source_name: str) -> List[Dict]:
    reader = PdfReader(filelike)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        txt = " ".join(txt.split())
        pages.append({"source": source_name, "page": i, "text": txt})
    return pages

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def build_corpus_from_pdfs(pdf_dir: str = PDF_DIR) -> List[Dict]:
    corpus = []
    for fname in sorted(os.listdir(pdf_dir)):
        if not fname.lower().endswith(".pdf"):
            continue
        fpath = os.path.join(pdf_dir, fname)
        with open(fpath, "rb") as f:
            pages = extract_text_from_pdf(f, fname)
        for p in pages:
            for ch in chunk_text(p["text"]):
                corpus.append({"source": p["source"], "page": p["page"], "text": ch})
    return corpus

def embed_texts(embedder: SentenceTransformer, texts: List[str]) -> np.ndarray:
    embs = embedder.encode(
        texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
    )
    return embs.astype("float32")

def save_index(index: faiss.Index, metas: List[Dict], dim: int, embedder_name: str):
    faiss.write_index(index, INDEX_PATH)
    meta = {"dim": dim, "embedder": embedder_name, "metas": metas}
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_index():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        raise RuntimeError(
            "Vector store não encontrado. Vai ao sidebar e clica em 'Build/Rebuild Index' depois de adicionares PDFs."
        )
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta["metas"], meta

def build_or_rebuild_index(embedder_name: str = DEFAULT_EMBEDDER):
    ensure_dirs()
    corpus = build_corpus_from_pdfs(PDF_DIR)
    if not corpus:
        raise RuntimeError("Nenhum PDF encontrado em ./pdfs")
    embedder = load_embedder(embedder_name)
    texts = [c["text"] for c in corpus]
    embs = embed_texts(embedder, texts)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    save_index(index, corpus, dim, embedder_name)

def retrieve(query: str, k: int = 8, embedder=None, index=None, metas=None, meta_info=None):
    if embedder is None or index is None or metas is None:
        index, metas, meta_info = load_index()
        embedder = load_embedder(meta_info.get("embedder", DEFAULT_EMBEDDER))
    q = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, k)
    hits = []
    for idx, sc in zip(ids[0], scores[0]):
        if idx == -1:
            continue
        m = metas[idx]
        hits.append({"score": float(sc), "source": m["source"], "page": m["page"], "text": m["text"]})
    return hits

def build_context(hits: List[Dict]):
    citation_map, ctx_parts, next_id = {}, [], 1
    for h in hits:
        key = f'{h["source"]}#p{h["page"]}'
        if key not in citation_map:
            citation_map[key] = next_id
            next_id += 1
        cid = citation_map[key]
        snippet = h["text"].strip().replace("\n", " ")
        ctx_parts.append(f"[{cid}] ({h['source']} p.{h['page']}) {snippet}")
    return "\n\n".join(ctx_parts), citation_map

def format_citations(citation_map: Dict[str, int]) -> str:
    return "\n".join([f"[{num}] {key.replace('#p', ' • p.')}" for key, num in sorted(citation_map.items(), key=lambda kv: kv[1])])

def stream_ollama_chat(model: str, system_prompt: str, user_prompt: str):
    payload = {
        "model": model,
        "stream": True,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    try:
        # Important fix: timeout=None (não 0)
        with requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, stream=True, timeout=None) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                if "message" in data and "content" in data["message"]:
                    yield data["message"]["content"]
                if data.get("done", False):
                    break
    except requests.exceptions.ConnectionError:
        yield "⚠️ Não consegui ligar ao Ollama. Garante que o serviço está a correr com `ollama serve` e que o modelo foi feito pull."
    except Exception as e:
        yield f"⚠️ Erro ao contactar o Ollama: {e}"

# ----------------------- Streamlit App -----------------------
st.set_page_config(page_title="Chat com Documentos (RAG)", page_icon="📄", layout="wide")
st.title(APP_TITLE)
ensure_dirs()

with st.sidebar:
    st.subheader("Setup")
    uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        for up in uploaded:
            with open(os.path.join(PDF_DIR, up.name), "wb") as f:
                f.write(up.getbuffer())
        st.success(f"Guardado(s): {', '.join([u.name for u in uploaded])}")
    embedder_name = st.text_input("Sentence-Transformers model", value=DEFAULT_EMBEDDER)
    if st.button("🔧 Build/Rebuild Index"):
        try:
            build_or_rebuild_index(embedder_name)
            st.success("Índice construído com sucesso ✅")
        except Exception as e:
            st.error(f"Falhou ao construir índice: {e}")
    st.subheader("Ollama")
    model_name = st.text_input("Modelo Ollama", value=DEFAULT_OLLAMA_MODEL)
    st.caption("Dica: noutro terminal corre `ollama serve` e faz `ollama pull gemma3:1b` (ou a tag que preferires).")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Olá! Carrega PDFs, constrói o índice e pergunta-me algo."}
    ]

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Escreve a tua pergunta…")

def answer_question(query: str):
    # 1) Garantir que temos índice
    try:
        index, metas, meta_info = load_index()
    except Exception as e:
        return f"⚠️ {e}", ""

    # 2) Garantir que o Ollama está (provavelmente) online
    if not ollama_is_up():
        return ("⚠️ Ollama parece offline. Abre um terminal e corre `ollama serve`. "
                "Depois faz `ollama pull` do modelo que definiste no sidebar."), ""

    # 3) Retrieval
    embedder = load_embedder(meta_info.get("embedder", DEFAULT_EMBEDDER))
    hits = retrieve(query, k=8, embedder=embedder, index=index, metas=metas, meta_info=meta_info)
    if not hits:
        return "Não encontrei contexto relevante nos PDFs.", ""

    context, cmap = build_context(hits)

    # 4) Prompting
    sys_prompt = (
        "You are an assistant that answers strictly using the provided CONTEXT.\n"
        "If the answer is not present in the context, say you don't know.\n"
        "Write concise answers in Portuguese (Portugal). Use inline numeric citations like [1], [2]."
    )
    user_prompt = (
        f"QUESTION:\n{query}\n\n"
        f"CONTEXT (sources with IDs):\n{context}\n\n"
        "Answer in Portuguese (Portugal) and include citations [n] for the specific sources you used."
    )

    # 5) Streaming
    stream = stream_ollama_chat(model_name, sys_prompt, user_prompt)
    collected, placeholder = "", st.empty()
    for token in stream:
        collected += token
        placeholder.markdown(collected)

    return collected, format_citations(cmap)

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        answer, cites = answer_question(question)
        st.markdown(answer)
        if cites:
            with st.expander("📚 Fontes"):
                st.markdown(cites)
    st.session_state["messages"].append({"role": "assistant", "content": answer})
