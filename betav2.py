import os
import gc
import time
import re
import traceback
import json
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
def init_session_state():
    keys_to_init = {
        'conversation': [],
        'conversation_memory': [],
        'processing': False,
        'db_exists': False,
        'uploaded_files': [],
        'model_name': "gemini-1.5-flash",
        'sources': set(),
        'first_load': True,
        'theme': 'dark',
        'error_count': 0,
        'total_queries': 0,
        'average_response_time': 0.0
    }
    for key, default in keys_to_init.items():
        if key not in st.session_state:
            st.session_state[key] = default

# Set page config FIRST
st.set_page_config(
    page_title="Ultimate CFA Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "CFA Assistant v5.0 • RAG System with Advanced Reasoning"
    }
)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("🔑 GEMINI_API_KEY not found in .env file or environment variables")
    st.stop()

# Enhanced Configuration
CONFIG = {
    "DB_PATH": "./cfa_chroma_db",
    "RAG_TOPK": 7,
    "CHUNK_SIZE": 1200,
    "CHUNK_OVERLAP": 200,
    "MAX_ANSWER_LENGTH": 4000,
    "LLM_TEMPERATURE": 0.3,
    "LLM_MAX_TOKENS": 3000,
    "MEMORY_SIZE": 5,
    "MAX_RETRIES": 3,
    "TIMEOUT_SECONDS": 30,
    "MIN_CHUNK_LENGTH": 100,
    "MAX_FILE_SIZE_MB": 50,
    "MAX_TEXT_LENGTH": 800000
}

# Custom CSS for high contrast dark theme
st.markdown("""
<style>
    /* Global Styles */
    :root {
        --primary: #1e88e5;
        --secondary: #64b5f6;
        --background: #111827;
        --card: #1f2937;
        --text-primary: #e5e7eb;
        --text-secondary: #9ca3af;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --accent: #8b5cf6;
        --border: #374151;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: var(--background) !important;
        color: var(--text-primary) !important;
    }
    
    /* Header styles */
    .header-container {
        background: linear-gradient(90deg, var(--accent), var(--primary));
        padding: 1.5rem 1rem;
        border-radius: 0 0 30px 30px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px 0 rgba(0,0,0,0.3);
    }
    
    .header-title {
        color: white !important;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    
    /* Chat containers */
    .stChatMessage {
        background: var(--card) !important;
        border-radius: 18px !important;
        border: none !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        margin: 1rem 0;
        padding: 1.5rem 1.25rem;
    }
    
    /* User message bubble */
    [data-testid="stChatMessage-user"] {
        background: #1f4068 !important;
        border: 1px solid var(--primary) !important;
    }
    
    /* Assistant message bubble */
    [data-testid="stChatMessage-assistant"] {
        background: linear-gradient(135deg, #1f2937, #111827) !important;
        border: 1px solid var(--accent) !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--card) !important;
        border-right: 1px solid var(--border) !important;
    }
    
    .sidebar-section {
        padding: 1rem;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background-color: var(--success);
        margin-right: 8px;
    }
    
    .status-warning {
        background-color: var(--warning) !important;
    }
    
    /* Buttons */
    .stButton>button {
        border: none !important;
        background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(30, 136, 229, 0.4);
    }
    
    /* Input fields */
    .stTextInput>div>div>input {
        background: var(--card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
    }
    
    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        color: var(--secondary);
        margin-bottom: 1.5rem;
        font-weight: 600;
        font-size: 1.25rem;
    }
    
    .section-header svg {
        margin-right: 10px;
        font-size: 1.5rem;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: var(--accent) !important;
    }
    
    /* Badges and chips */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        background: var(--accent);
        color: white;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0 5px;
    }
    
    /* Expanders */
    .stExpander {
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        margin: 1rem 0;
    }
    
    .stExpander summary {
        color: var(--secondary) !important;
        font-weight: 600 !important;
        background: rgba(30, 41, 59, 0.7) !important;
        border-radius: 12px 12px 0 0 !important;
        padding: 1rem 1.25rem !important;
    }
    
    /* File uploader */
    .uploader-container {
        background: rgba(39, 50, 78, 0.5) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
init_session_state()

# Enhanced Helper Functions
def create_file_path(filename: str) -> str:
    """Create safe file path"""
    try:
        os.makedirs(CONFIG["DB_PATH"], exist_ok=True)
        safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
        return os.path.join(CONFIG["DB_PATH"], safe_filename)
    except Exception as e:
        logger.error(f"Error creating file path: {e}")
        return f"./cfa_chroma_db/{filename}"

def validate_file(uploaded_file) -> Tuple[bool, str]:
    """Validate uploaded file"""
    if not uploaded_file:
        return False, "No file provided"
    
    if not uploaded_file.name.lower().endswith('.pdf'):
        return False, "Only PDF files are supported"
    
    file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
    if file_size_mb > CONFIG["MAX_FILE_SIZE_MB"]:
        return False, f"File too large ({file_size_mb:.1f}MB). Maximum size: {CONFIG['MAX_FILE_SIZE_MB']}MB"
    
    return True, "Valid file"

def update_conversation_memory(question: str, answer: str, sources: set):
    """Maintain recent conversation history"""
    memory_entry = {
        'question': question,
        'answer': answer,
        'sources': list(sources),
        'timestamp': datetime.now().isoformat()
    }
    
    st.session_state.conversation_memory.append(memory_entry)
    
    if len(st.session_state.conversation_memory) > CONFIG["MEMORY_SIZE"]:
        st.session_state.conversation_memory = st.session_state.conversation_memory[-CONFIG["MEMORY_SIZE"]:]

def get_conversation_context() -> str:
    """Generate context from history"""
    if not st.session_state.conversation_memory:
        return ""
    
    context_parts = ["RECENT CONVERSATION HISTORY:"]
    for i, entry in enumerate(st.session_state.conversation_memory[-3:], 1):
        context_parts.append(f"\nQ{i}: {entry['question'][:200]}...")
        context_parts.append(f"A{i}: {entry['answer'][:300]}...")
    
    context_parts.append("\nCURRENT QUESTION:")
    return "\n".join(context_parts)

def safe_extract_text(pdf_path: str) -> str:
    """Robust text extraction"""
    for attempt in range(CONFIG["MAX_RETRIES"]):
        try:
            # Try PyPDF first
            try:
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                text = " ".join([page.page_content for page in pages])
                if text.strip() and len(text) > CONFIG["MIN_CHUNK_LENGTH"]:
                    logger.info(f"Extracted text using PyPDF: {len(text)} chars")
                    return text
            except Exception as e:
                logger.warning(f"PyPDF extraction failed (#{attempt}): {str(e)}")
            
            # Fallback to Unstructured
            try:
                loader = UnstructuredPDFLoader(pdf_path)
                data = loader.load()
                if data and data[0].page_content:
                    text = data[0].page_content
                    if len(text) > CONFIG["MIN_CHUNK_LENGTH"]:
                        logger.info(f"Extracted text using Unstructured: {len(text)} chars")
                        return text
            except Exception as e:
                logger.warning(f"Unstructured extraction failed (#{attempt}): {str(e)}")
            
            time.sleep(1)  # Wait before retry
        except Exception as e:
            logger.error(f"Text extraction error: {traceback.format_exc()}")
            break
            
    return ""

def preprocess_text(text: str) -> str:
    """Prepare text for processing"""
    if not text:
        return ""
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\'\/\%\$\@\#\&\*\+\=\<\>\~\`]', ' ', text)
    text = re.sub(r'(\d+\.)\s*([A-Z])', r'\1 \2', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    return text.strip()

def process_pdf_chunk(pdf_path: str, filename: str, vector_db) -> int:
    """Process a PDF and add chunks to vector store"""
    try:
        # Extract text
        raw_text = safe_extract_text(pdf_path)
        if not raw_text.strip():
            st.error(f"⛔ Failed to extract text from: {filename}")
            return 0
        
        # Preprocess
        text = preprocess_text(raw_text)
        
        # Limit text size
        max_chars = CONFIG["MAX_TEXT_LENGTH"]
        if len(text) > max_chars:
            text = text[:max_chars]
            logger.warning(f"Text truncated to {max_chars} chars for {filename}")
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["CHUNK_SIZE"],
            chunk_overlap=CONFIG["CHUNK_OVERLAP"],
            length_function=len,
            separators=["\n\n", "\n", ". ", " "]
        )
        
        chunks = splitter.split_text(text)
        valid_chunks = [chunk for chunk in chunks if len(chunk.strip()) >= CONFIG["MIN_CHUNK_LENGTH"]]
        
        if not valid_chunks:
            st.error(f"⛔ No valid chunks generated for: {filename}")
            return 0
        
        # Add to vector store
        try:
            metadatas = [{"source": filename} for _ in valid_chunks]
            vector_db.add_texts(texts=valid_chunks, metadatas=metadatas)
            return len(valid_chunks)
        except Exception as e:
            logger.error(f"Error adding chunks: {str(e)}")
            return 0
            
    except Exception as e:
        logger.error(f"Error processing {filename}: {traceback.format_exc()}")
        return 0

def build_knowledge_base(uploaded_files):
    """Build or update knowledge base"""
    if not uploaded_files:
        st.warning("No files provided")
        return 0
    
    try:
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=API_KEY,
            task_type="retrieval_document"
        )
        
        # Initialize Chroma
        vector_db = Chroma(
            persist_directory=CONFIG["DB_PATH"],
            embedding_function=embeddings,
            collection_name="cfa_knowledge"
        )
        
        total_chunks = 0
        source_files = set()
        progress_bar = st.progress(0)
        status_container = st.empty()
        
        # Process files
        for i, uploaded_file in enumerate(uploaded_files):
            file_path = create_file_path(uploaded_file.name)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            start = time.time()
            chunks_added = process_pdf_chunk(file_path, uploaded_file.name, vector_db)
            proc_time = time.time() - start
            
            if chunks_added > 0:
                total_chunks += chunks_added
                source_files.add(uploaded_file.name)
                st.session_state.uploaded_files.append(uploaded_file.name)
                status_container.success(f"✅ {uploaded_file.name}: Added {chunks_added} chunks ({proc_time:.1f}s)")
            else:
                status_container.warning(f"⚠️ {uploaded_file.name}: Processing failed")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
            gc.collect()
            time.sleep(0.1)
        
        # Finalize
        vector_db.persist()
        st.session_state.db_exists = True
        st.session_state.sources = source_files
        progress_bar.empty()
        
        if total_chunks > 0:
            status_container.success(f"✅ Knowledge base updated! Added {total_chunks} chunks across {len(source_files)} files")
            time.sleep(2)
            st.balloons()
            st.session_state.conversation = []  # Reset conversation for new knowledge base
        else:
            status_container.error("❌ No content was added to knowledge base")
        
        return total_chunks
    
    except Exception as e:
        logger.error(f"Knowledge base build failed: {traceback.format_exc()}")
        st.error(f"❌ Critical system error: {str(e)}")
        return 0

def modern_reasoning_engine(question: str, context: str, conversation_history: str = "") -> str:
    """Generate response using AI"""
    SYSTEM_PROMPT = """[System] As a CFA expert, provide detailed, accurate answers using ONLY the documents provided. 
Structure your response:
1. Identify the CFA level and topic
2. Define key terms
3. Provide step-by-step explanation
4. Include formulas where relevant
5. Summarize key points

[Conversation History]
{conversation_context}

[Relevant Documents]
{context}

[Question]
{question}"""

    try:
        # Initialize model
        llm = ChatGoogleGenerativeAI(
            model=st.session_state.model_name,
            google_api_key=API_KEY,
            temperature=CONFIG["LLM_TEMPERATURE"],
            max_output_tokens=CONFIG["LLM_MAX_TOKENS"]
        )
        
        # Format prompt
        formatted_prompt = SYSTEM_PROMPT.format(
            conversation_context=conversation_history[:5000] if conversation_history else "[No conversation history]",
            context=context[:25000],
            question=question[:1000]
        )
        
        # Generate response
        response = llm.invoke(formatted_prompt)
        answer = response.content[:CONFIG["MAX_ANSWER_LENGTH"]]
        return answer
        
    except Exception as e:
        logger.error(f"Reasoning error: {traceback.format_exc()}")
        return "⚠️ Failed to generate answer. Please try again."

def retrieve_context(question: str) -> Tuple[str, set]:
    """Retrieve relevant context from knowledge base"""
    for attempt in range(CONFIG["MAX_RETRIES"]):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=API_KEY
            )
            
            vector_db = Chroma(
                persist_directory=CONFIG["DB_PATH"],
                embedding_function=embeddings,
                collection_name="cfa_knowledge"
            )
            
            docs = vector_db.similarity_search(question, k=CONFIG["RAG_TOPK"])
            
            if not docs:
                return "No relevant documents found", set()
            
            context_parts = []
            sources = set()
            
            for i, doc in enumerate(docs):
                source = doc.metadata.get("source", "Unknown")
                sources.add(source)
                context_parts.append(f"[Document {i+1} - {source}]\n{doc.page_content[:1000]}\n")
            
            return "\n".join(context_parts), sources
            
        except Exception as e:
            logger.warning(f"Retrieval attempt #{attempt+1} failed: {str(e)}")
            time.sleep(1)
    
    return "Unable to retrieve documents", set()

def process_query(question: str) -> Tuple[str, set, float]:
    """Handle user query"""
    start = time.time()
    
    try:
        # Update stats
        st.session_state.total_queries += 1
        
        # Validate
        if not question.strip() or len(question.strip()) < 3:
            return "⚠️ Please enter a valid question (minimum 3 characters)", set(), 0
        
        # Check knowledge base
        if not st.session_state.db_exists:
            return "⚠️ Please build the knowledge base first by uploading PDFs", set(), 0
        
        # Get context
        context, sources = retrieve_context(question)
        if not sources:
            return "⚠️ No relevant information found in knowledge base", set(), 0
        
        # Get conversation context
        conversation_context = get_conversation_context()
        
        # Generate answer
        answer = modern_reasoning_engine(question, context, conversation_context)
        
        # Update memory
        update_conversation_memory(question, answer, sources)
        
        # Update performance metrics
        proc_time = time.time() - start
        curr_avg = st.session_state.average_response_time
        total = st.session_state.total_queries
        st.session_state.average_response_time = ((curr_avg * (total - 1)) + proc_time) / total
        
        return answer, sources, proc_time
        
    except Exception as e:
        logger.error(f"Query processing error: {traceback.format_exc()}")
        return "⚠️ An unexpected error occurred. Please try again.", set(), 0

def render_message(role: str, content: str, sources: set = set()):
    """Display message in chat"""
    with st.chat_message(role, avatar="💼" if role == "assistant" else "🧑‍🎓"):
        st.markdown(content)
        if sources and role == "assistant":
            with st.expander(f"📚 Source Documents ({len(sources)})"):
                for i, source in enumerate(sorted(sources), 1):
                    st.write(f"{i}. {source}")

def render_system_status():
    """Show system health in sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.subheader("📊 Performance")
        st.metric("Total Queries", st.session_state.total_queries)
        if st.session_state.total_queries > 0:
            st.metric("Avg. Response", f"{st.session_state.average_response_time:.2f}s")
        st.metric("Errors", st.session_state.error_count)
        
        health_status = "🟢 Healthy" if st.session_state.error_count < 2 else "🟡 Warning" if st.session_state.error_count < 5 else "🔴 Critical"
        st.metric("System Status", health_status)

# Main Application
def main():
    # Header
    with st.container():
        st.markdown("""
        <div class="header-container">
            <h1 class="header-title">📊 Ultimate CFA Assistant</h1>
            <p style="color: #e0e0e0; font-size: 1.2rem;">AI-powered CFA exam preparation with RAG technology</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar - Knowledge Management
    with st.sidebar:
        st.markdown("### 🧠 Knowledge Base")
        st.markdown("Upload your CFA curriculum materials:")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Select PDF Files",
            type="pdf",
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        # Build KB button
        if st.button("🛠️ Build Knowledge Base", use_container_width=True, type="primary"):
            if not uploaded_files:
                st.warning("Please upload files first")
            else:
                with st.spinner("Processing documents..."):
                    build_knowledge_base(uploaded_files)
        
        # Reset option
        if st.button("🗑️ Clear Conversation", help="Start a new conversation", use_container_width=True):
            st.session_state.conversation = []
            st.success("Conversation cleared")
            st.rerun()
        
        # System info
        render_system_status()
        
        # Uploaded files list
        if st.session_state.uploaded_files:
            with st.expander("📚 Uploaded Materials", expanded=True):
                st.write(f"Loaded {len(st.session_state.uploaded_files)} files:")
                for file in st.session_state.uploaded_files[:5]:
                    st.caption(f"📄 {file}")
                if len(st.session_state.uploaded_files) > 5:
                    st.caption(f"Plus {len(st.session_state.uploaded_files)-5} more files")

    # Main content - Chat interface
    with st.container():
        # Display conversation
        for msg in st.session_state.conversation:
            render_message(msg["role"], msg["content"], msg.get("sources", set()))
        
        # User input
        query = st.chat_input(
            "Ask about CFA concepts, formulas, or topics...", 
            disabled=st.session_state.processing or not st.session_state.db_exists
        )
        
        if query and not st.session_state.processing:
            # Add to conversation
            st.session_state.conversation.append({"role": "user", "content": query})
            st.session_state.processing = True
            
            # Show thinking status
            with st.chat_message("assistant", avatar="💼"):
                thinking_placeholder = st.empty()
                thinking_placeholder.markdown("💭 Processing your question...")
                
                try:
                    # Generate response
                    answer, sources, time_spent = process_query(query)
                    
                    thinking_placeholder.empty()
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander(f"📚 Source Documents ({len(sources)})", expanded=False):
                            for i, source in enumerate(sorted(sources), 1):
                                st.write(f"{i}. {source}")
                    
                    st.caption(f"Generated in {time_spent:.2f} seconds")
                    
                    # Update conversation
                    st.session_state.conversation.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    error = f"Error: {str(e)}"
                    st.error(error)
                    st.session_state.conversation.append({
                        "role": "assistant",
                        "content": f"⚠️ {error}",
                        "sources": set()
                    })
                    st.session_state.error_count += 1
            
            st.session_state.processing = False

# Run the app
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Critical error: {str(e)}")
        logger.critical(f"Application crashed: {traceback.format_exc()}")
