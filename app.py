"""
aiQS - AI Document Question Answering System
Minimalist monochrome dashboard (pure black, white, and subtle grays).
"""

import os
import streamlit as st
from dotenv import load_dotenv

from document_loader import load_document, chunk_document_pages
from rag_engine import RAGEngine, DEFAULT_LLM_MODEL

# Load .env variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="aiQS",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Strict Monochrome CSS: White, Black, Neutral Grays only
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #111111;
    }

    /* Primary buttons */
    button[kind="primary"] {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 1px solid #000000 !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }
    button[kind="primary"]:hover {
        background-color: #262626 !important;
        border-color: #262626 !important;
        color: #FFFFFF !important;
    }

    /* Secondary buttons */
    button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #111111 !important;
        border: 1px solid #E5E5E5 !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }
    button[kind="secondary"]:hover {
        background-color: #F5F5F5 !important;
        border-color: #D4D4D4 !important;
        color: #000000 !important;
    }

    /* Clean Monochrome Header */
    .dash-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 0 18px 0;
        border-bottom: 1px solid #E5E5E5;
        margin-bottom: 20px;
    }
    .dash-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #000000;
        margin: 0;
        letter-spacing: -0.3px;
    }
    .dash-subtitle {
        font-size: 0.85rem;
        color: #666666;
        margin: 2px 0 0 0;
    }
    .dash-status {
        font-size: 0.8rem;
        font-weight: 600;
        color: #111111;
        background: #F5F5F5;
        border: 1px solid #E5E5E5;
        padding: 4px 10px;
        border-radius: 6px;
    }

    /* Minimalist Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 8px;
        padding: 14px 16px;
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #737373;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #000000;
        margin-top: 4px;
        letter-spacing: -0.5px;
    }

    /* Monochrome Citations */
    .citation-card {
        background: #FAFAFA;
        border: 1px solid #E5E5E5;
        border-left: 3px solid #000000;
        border-radius: 6px;
        padding: 12px 14px;
        margin-top: 8px;
        margin-bottom: 10px;
    }
    .citation-meta {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-mono {
        background: #E5E5E5;
        color: #000000;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .badge-mono-border {
        background: #FFFFFF;
        color: #333333;
        border: 1px solid #E5E5E5;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .citation-text {
        font-size: 0.85rem;
        color: #333333;
        line-height: 1.5;
    }

    /* Sidebar Clean Title */
    .sidebar-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #000000;
        margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if "rag_engine" not in st.session_state:
    env_key = os.getenv("GEMINI_API_KEY", "")
    st.session_state.rag_engine = RAGEngine(api_key=env_key)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []


# ==============================================================================
# SIDEBAR (BLACK & WHITE MINIMALIST)
# ==============================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">Document Manager</div>', unsafe_allow_html=True)

    # If API key is missing, unobtrusive input
    if not st.session_state.rag_engine.is_configured():
        api_key_val = st.text_input(
            "API Key",
            type="password",
            placeholder="Enter Gemini API Key...",
            help="Get your key at https://aistudio.google.com/"
        )
        if api_key_val:
            st.session_state.rag_engine.set_api_key(api_key_val)
            with open(".env", "a", encoding="utf-8") as f:
                f.write(f"\nGEMINI_API_KEY={api_key_val.strip()}\n")
            st.rerun()
        st.caption("Enter API key to enable questioning.")
        st.divider()

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "doc", "txt", "md", "csv", "json"],
        accept_multiple_files=True,
        help="Upload PDF, Word, or text files to build your knowledge base."
    )

    index_btn = st.button("⬆ Upload & Index Files", use_container_width=True, type="primary")
    sample_btn = st.button("📄 Load Sample Syllabus", use_container_width=True)

    # Sample Document Handler
    if sample_btn:
        sample_path = os.path.join(os.path.dirname(__file__), "sample_documents", "MCA_Semester_1_Syllabus.txt")
        if os.path.exists(sample_path):
            with st.spinner("Indexing sample syllabus..."):
                try:
                    pages = load_document(sample_path, "MCA_Semester_1_Syllabus.txt")
                    chunks = chunk_document_pages(pages, chunk_size=700, chunk_overlap=150)
                    st.session_state.rag_engine.index_chunks(chunks, session_id="default")
                    st.session_state.indexed_docs = [{
                        "name": "MCA_Semester_1_Syllabus.txt",
                        "pages": len(pages),
                        "chunks": len(chunks)
                    }]
                    st.toast("Sample syllabus indexed.", icon="✓")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # Uploaded Files Handler
    if index_btn:
        if not uploaded_files:
            st.warning("Please browse or drag files first using the box above.")
        elif not st.session_state.rag_engine.is_configured():
            st.error("API key is required.")
        else:
            with st.spinner(f"Uploading & indexing {len(uploaded_files)} document(s)..."):
                try:
                    all_chunks = []
                    new_docs = []
                    for uf in uploaded_files:
                        pages = load_document(uf, uf.name)
                        chunks = chunk_document_pages(pages, chunk_size=700, chunk_overlap=150)
                        all_chunks.extend(chunks)
                        new_docs.append({
                            "name": uf.name,
                            "pages": len(pages),
                            "chunks": len(chunks)
                        })

                    if all_chunks:
                        count = st.session_state.rag_engine.index_chunks(all_chunks, session_id="default")
                        st.session_state.indexed_docs.extend(new_docs)
                        st.toast(f"Successfully indexed {count} chunks across {len(new_docs)} file(s).", icon="✓")
                        st.rerun()
                    else:
                        st.warning("No readable text found in documents.")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Indexed documents list & Reset
    if st.session_state.indexed_docs:
        st.divider()
        st.markdown("**Active Indexed Documents**")
        for d in st.session_state.indexed_docs:
            st.markdown(f"• **{d['name']}** ({d['chunks']} chunks, {d['pages']} pgs)")

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        if st.button("Clear All Documents", use_container_width=True):
            st.session_state.rag_engine.clear_index()
            st.session_state.chat_history = []
            st.session_state.indexed_docs = []
            try:
                from database import delete_chat_history, delete_documents_for_session, delete_chunks_for_session
                delete_chat_history("default")
                delete_documents_for_session("default")
                delete_chunks_for_session("default")
            except Exception:
                pass
            st.rerun()



# ==============================================================================
# MAIN WORKSPACE (BLACK & WHITE)
# ==============================================================================

stats = st.session_state.rag_engine.get_stats()
status_label = "Ready" if stats["total_chunks"] > 0 else "No Documents"

st.markdown(f"""
<div class="dash-header">
    <div>
        <h1 class="dash-title">Document Question Answering</h1>
        <p class="dash-subtitle">Grounded search and answer synthesis based strictly on uploaded files.</p>
    </div>
    <div class="dash-status">
        Status: {status_label}
    </div>
</div>
""", unsafe_allow_html=True)

# Metric Ribbon (Black & White)
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Documents</div>
        <div class="metric-val">{stats['total_sources']}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Indexed Chunks</div>
        <div class="metric-val">{stats['total_chunks']}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Retrieval</div>
        <div class="metric-val">Top-4 Chunks</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# Tabs (Q&A and Documents)
tab_qa, tab_docs = st.tabs(["Question & Answer", "Documents"])

# ------------------------------------------------------------------------------
# TAB 1: QUESTION & ANSWER
# ------------------------------------------------------------------------------
with tab_qa:
    if stats["total_chunks"] == 0:
        st.info("Upload documents or click 'Sample' in the sidebar to begin.")
    else:
        # Suggested questions
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            if st.button("Subjects in Semester 1?", use_container_width=True):
                st.session_state.quick_query = "What are the subjects in Semester 1?"
        with col_s2:
            if st.button("Topics in Unit 3?", use_container_width=True):
                st.session_state.quick_query = "Explain the topics covered in Unit 3 in detail."
        with col_s3:
            if st.button("Examination scheme?", use_container_width=True):
                st.session_state.quick_query = "What is the examination evaluation scheme?"

    # Render Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander(f"Sources ({len(msg['sources'])} citations)"):
                    for s in msg["sources"]:
                        score_pct = int(s["score"] * 100)
                        st.markdown(f"""
                        <div class="citation-card">
                            <div class="citation-meta">
                                <span class="badge-mono">{s['source_file']}</span>
                                <span class="badge-mono-border">Page {s['page_number']}</span>
                                <span class="badge-mono-border">Match: {score_pct}%</span>
                            </div>
                            <div class="citation-text">
                                "{s['text']}"
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    # Chat Input
    query_text = None
    if "quick_query" in st.session_state and st.session_state.quick_query:
        query_text = st.session_state.quick_query
        st.session_state.quick_query = None
    else:
        query_text = st.chat_input("Ask a question about your documents...")

    if query_text:
        if not st.session_state.rag_engine.is_configured():
            st.error("API key is required in the sidebar.")
        elif stats["total_chunks"] == 0:
            st.warning("Upload a document or click 'Sample' in the sidebar first.")
        else:
            # User Message
            st.session_state.chat_history.append({"role": "user", "content": query_text})
            with st.chat_message("user"):
                st.markdown(query_text)

            # Assistant Generation
            with st.chat_message("assistant"):
                with st.spinner("Searching context..."):
                    retrieved = st.session_state.rag_engine.retrieve_context(query_text, top_k=4)
                    stream = st.session_state.rag_engine.generate_grounded_response_stream(
                        question=query_text,
                        retrieved_chunks=retrieved,
                        model_name=DEFAULT_LLM_MODEL
                    )
                    full_response = st.write_stream(stream)

                sources_meta = []
                if retrieved:
                    for chunk, score in retrieved:
                        sources_meta.append({
                            "source_file": chunk.source_file,
                            "page_number": chunk.page_number,
                            "score": score,
                            "text": chunk.text
                        })

                    with st.expander(f"Sources ({len(retrieved)} citations)"):
                        for s in sources_meta:
                            score_pct = int(s["score"] * 100)
                            st.markdown(f"""
                            <div class="citation-card">
                                <div class="citation-meta">
                                    <span class="badge-mono">{s['source_file']}</span>
                                    <span class="badge-mono-border">Page {s['page_number']}</span>
                                    <span class="badge-mono-border">Match: {score_pct}%</span>
                                </div>
                                <div class="citation-text">
                                    "{s['text']}"
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources_meta
                })


# ------------------------------------------------------------------------------
# TAB 2: DOCUMENTS
# ------------------------------------------------------------------------------
with tab_docs:
    st.markdown("### Uploaded Documents")
    if not st.session_state.indexed_docs:
        st.info("No documents uploaded yet.")
    else:
        for doc in st.session_state.indexed_docs:
            col_d1, col_d2, col_d3 = st.columns([3, 1, 1])
            with col_d1:
                st.markdown(f"**`{doc['name']}`**")
            with col_d2:
                st.markdown(f"Pages: {doc['pages']}")
            with col_d3:
                st.markdown(f"Chunks: {doc['chunks']}")
            st.divider()

        st.markdown("#### Chunk Content Preview")
        preview_chunks = st.session_state.rag_engine.vector_store.chunks[:10]
        if preview_chunks:
            selected_chunk = st.selectbox(
                "Select chunk ID to view:",
                options=[c.chunk_id for c in preview_chunks]
            )
            for c in preview_chunks:
                if c.chunk_id == selected_chunk:
                    st.text_area("Content", c.text, height=180, disabled=True)
                    st.caption(f"File: {c.source_file} | Page: {c.page_number}")

# Vercel ASGI Serverless export
try:
    from api.index import app as handler
    app = handler
except Exception:
    pass

