# ================== app.py ==================

# ---- torch.classes workaround ----
import sys, types
if "torch.classes" not in sys.modules:
    sys.modules["torch.classes"] = types.ModuleType("torch.classes")

# ---- load env ----
from dotenv import load_dotenv
load_dotenv()

# ---- imports ----
import streamlit as st
import chromadb
from chromadb.api.types import EmbeddingFunction
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

import camelot
import tabula
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path
import pandas as pd

import uuid, os, re, tempfile

# ================== CONFIG ==================
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    st.error("❌ NVIDIA_API_KEY not set (check .env)")
    st.stop()

# Using NVIDIA's reasoning model for better analysis
REASONING_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"
EMBED_MODEL = "nvidia/nv-embed-v1"
DIST_THRESHOLD = 1.2

# ================== LLM ==================
@st.cache_resource
def get_llm():
    return ChatNVIDIA(
        model=REASONING_MODEL,
        api_key=NVIDIA_API_KEY,
        temperature=0.2,  # Slight creativity for better explanations
        max_tokens=2048   # Allow longer reasoning chains
    )

llm = get_llm()

# ================== EMBEDDINGS ==================
class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, embedder):
        self.embedder = embedder
    def __call__(self, input):
        return self.embedder.embed_documents(input)

@st.cache_resource
def get_embeddings():
    return NVIDIAEmbeddings(
        model=EMBED_MODEL,
        api_key=NVIDIA_API_KEY
    )

embeddings = get_embeddings()

# ================== CHROMA ==================
@st.cache_resource
def get_chroma():
    client = chromadb.Client()
    return client.get_or_create_collection(
        name="docs_collection",
        embedding_function=CustomEmbeddingFunction(embeddings)
    )

collection = get_chroma()

# ================== BM25 ==================
if 'bm25' not in st.session_state:
    st.session_state.bm25 = None
    st.session_state.bm25_docs = []
    st.session_state.bm25_metas = []

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

# ================== PDF EXTRACTION ==================
def extract_text_pages(pdf):
    reader = PdfReader(pdf)
    pages = []
    for i, p in enumerate(reader.pages):
        t = p.extract_text()
        if t:
            pages.append({"page": i+1, "text": t})
    return pages

def extract_tables_camelot(pdf_path):
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
        return [t.df for t in tables if t.df.shape[1] > 1]
    except Exception as e:
        st.warning(f"Camelot extraction failed: {e}")
        return []

def extract_tables_tabula(pdf_path):
    try:
        dfs = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True)
        return [df for df in dfs if not df.empty]
    except Exception as e:
        st.warning(f"Tabula extraction failed: {e}")
        return []

def extract_chart_text_ocr(pdf_path):
    texts = []
    try:
        pages = convert_from_path(pdf_path, dpi=300)
        for img in pages:
            img = np.array(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            if any(c.isdigit() for c in text):
                texts.append(text)
    except Exception as e:
        st.warning(f"OCR extraction failed: {e}")
    return texts

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    return splitter.split_text(text)

# ================== INGEST ==================
def ingest_pdf_once(pdf):
    if "indexed" not in st.session_state:
        st.session_state.indexed = set()

    if pdf.name in st.session_state.indexed:
        return "Already indexed ✔️"

    docs, metas, ids = [], [], []

    # Save uploaded file to temp location for table/OCR extraction
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf.read())
        tmp_path = tmp_file.name
    
    # Reset file pointer for text extraction
    pdf.seek(0)

    # --- Text pages ---
    pages = extract_text_pages(pdf)
    for p in pages:
        for i, chunk in enumerate(split_text(p["text"])):
            meta = {"source": pdf.name, "page": p["page"], "chunk_id": i}
            docs.append(chunk)
            metas.append(meta)
            ids.append(str(uuid.uuid4()))
            st.session_state.bm25_docs.append(tokenize(chunk))
            st.session_state.bm25_metas.append(meta)

    # --- Tables ---
    tables = extract_tables_camelot(tmp_path)
    if not tables:
        tables = extract_tables_tabula(tmp_path)

    for i, df in enumerate(tables):
        text = df.to_string(index=False)
        meta = {"source": pdf.name, "page": "table", "chunk_id": f"table_{i}"}
        docs.append(text)
        metas.append(meta)
        ids.append(str(uuid.uuid4()))
        st.session_state.bm25_docs.append(tokenize(text))
        st.session_state.bm25_metas.append(meta)

    # --- OCR charts ---
    ocr_texts = extract_chart_text_ocr(tmp_path)
    for i, t in enumerate(ocr_texts):
        meta = {"source": pdf.name, "page": "chart", "chunk_id": f"ocr_{i}"}
        docs.append(t)
        metas.append(meta)
        ids.append(str(uuid.uuid4()))
        st.session_state.bm25_docs.append(tokenize(t))
        st.session_state.bm25_metas.append(meta)

    # Clean up temp file
    os.unlink(tmp_path)

    # Add to collection
    if docs:
        collection.add(documents=docs, metadatas=metas, ids=ids)
        st.session_state.bm25 = BM25Okapi(st.session_state.bm25_docs)

    st.session_state.indexed.add(pdf.name)
    return f"Indexed {len(docs)} chunks (text + tables + charts) ✅"

# ================== HYBRID SEARCH ==================
def hybrid_search(query, k=5):  # Increased k for reasoning model
    vec = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    merged = {}

    for d, m, dist in zip(vec["documents"][0], vec["metadatas"][0], vec["distances"][0]):
        key = f"{m['page']}_{m['chunk_id']}"
        merged[key] = {"doc": d, "meta": m, "score": 1/(1+dist)}

    if st.session_state.bm25:
        scores = st.session_state.bm25.get_scores(tokenize(query))
        top = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        for idx, s in top:
            m = st.session_state.bm25_metas[idx]
            key = f"{m['page']}_{m['chunk_id']}"
            merged[key] = merged.get(key, {
                "doc": " ".join(st.session_state.bm25_docs[idx]), 
                "meta": m, 
                "score": 0
            })
            merged[key]["score"] += s

    return sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:k]

# ================== ANSWER WITH REASONING ==================
def answer_question(q):
    results = hybrid_search(q)
    if not results:
        return "Not found in the document.", None

    # Financial query validation
    if any(w in q.lower() for w in ["total", "income", "revenue", "ebitda", "pbt", "profit", "loss", "margin"]):
        if not any(any(c.isdigit() for c in r["doc"]) for r in results):
            return "Not found in the document.", None

    context, cites = [], []
    for r in results:
        tag = f"[p{r['meta']['page']}:c{r['meta']['chunk_id']}]"
        context.append(f"{tag} {r['doc']}")
        cites.append(tag)

    # Enhanced prompt for reasoning model
    prompt = f"""You are a  document analyst. Use ONLY the context provided below to answer the question.

CRITICAL RULES:
1. If the answer is not in the context, respond ONLY with: "Not found in the document."
2. If found, provide a clear, analytical answer with reasoning
3. For numerical data, cite exact figures from the context
4. For analytical questions, explain your reasoning step-by-step
5. Never make assumptions or add information not in the context

Context:
{chr(10).join(context)}

Question: {q}

Think through this step-by-step:
1. What information does the context provide?
2. Does it directly answer the question?
3. What's the complete answer?

Answer:"""

    ans = llm.invoke(prompt).content.strip()
    
    # Add sources if answer found
    if "not found in the document" not in ans.lower():
        ans += f"\n\n**Sources:** {', '.join(sorted(set(cites)))}"
    
    return ans, results

# ================== UI ==================
st.set_page_config("Document RAG", layout="wide")

# Header with model info
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📊Document RAG Assistant")
    st.caption("Hybrid RAG • Tables & Charts • Reasoning Model")
with col2:
    st.info(f"🧠 Model:\n{REASONING_MODEL.split('/')[-1]}")

uploaded = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded:
    with st.spinner("Indexing PDF..."):
        result = ingest_pdf_once(uploaded)
        st.success(result)

# Add example questions
with st.expander("💡 Example Questions"):
    st.markdown("""
    **Financial Analysis:**
    - What was the total revenue and how did it compare to last year?
    - Analyze the profit margins and explain any significant changes
    - What are the key financial risks mentioned?

    """)

if "chat" not in st.session_state:
    st.session_state.chat = []

# Display chat history
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
q = st.chat_input("Ask about the document...")
if q:
    st.session_state.chat.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    
    with st.chat_message("assistant"):
        with st.spinner("🧠 Reasoning..."):
            ans, debug = answer_question(q)
            st.markdown(ans)
            st.session_state.chat.append({"role": "assistant", "content": ans})

            if debug:
                with st.expander("🔍 Retrieved Evidence"):
                    for r in debug:
                        st.markdown(
                            f"**[p{r['meta']['page']}:c{r['meta']['chunk_id']}]** "
                            f"Relevance Score: {r['score']:.3f}"
                        )
                        st.text(r["doc"][:400] + ("..." if len(r["doc"]) > 400 else ""))
                        st.divider()

# Sidebar with controls
with st.sidebar:
    st.markdown("### 🎛️ Controls")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat = []
        st.rerun()
    
    if st.button("🔄 Reset Index"):
        st.session_state.indexed = set()
        st.session_state.bm25 = None
        st.session_state.bm25_docs = []
        st.session_state.bm25_metas = []
        st.success("Index reset! Upload a new PDF.")
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.markdown(f"""
    **Model:** NVIDIA Nemotron 70B
    
    **Features:**
    - Advanced reasoning capabilities
    - Step-by-step analysis
    - Financial document expertise
    - Hybrid search (semantic + keyword)
    - Table & chart extraction
    """)
    
    st.markdown("---")
    st.markdown("**Tips:**")
    st.markdown("""
    - Ask analytical questions for deeper insights
    - Request comparisons between metrics
    - Ask "why" and "how" questions
    - Request step-by-step explanations
    """)