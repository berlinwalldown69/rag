,import os
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
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state with enhanced memory management
def init_session_state():
    keys_to_init = {
        'conversation': [],
        'conversation_memory': [],  # Last 5 Q&A pairs for context
        'processing': False,
        'db_exists': False,
        'uploaded_files': [],
        'model_name': "gemini-1.5-flash",
        'sources': set(),
        'first_load': True,
        'system_health': {'status': 'healthy', 'last_check': datetime.now()},
        'error_count': 0,
        'total_queries': 0,
        'average_response_time': 0.0
    }
    for key, default in keys_to_init.items():
        if key not in st.session_state:
            st.session_state[key] = default

# Load environment variables with validation
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("🔑 GEMINI_API_KEY not found in .env file or environment variables")
    st.stop()

# Enhanced Configuration
CONFIG = {
    "DB_PATH": "./cfa_chroma_db",
    "RAG_TOPK": 7,  # Increased for better context
    "CHUNK_SIZE": 1200,  # Optimized chunk size
    "CHUNK_OVERLAP": 200,  # Better overlap for context continuity
    "MAX_ANSWER_LENGTH": 4000,  # Increased for detailed explanations
    "LLM_TEMPERATURE": 0.3,  # More focused responses
    "LLM_MAX_TOKENS": 3000,  # Increased token limit
    "MEMORY_SIZE": 5,  # Number of previous Q&A pairs to remember
    "MAX_RETRIES": 3,  # Retry mechanism for failed operations
    "TIMEOUT_SECONDS": 30,  # Request timeout
    "MIN_CHUNK_LENGTH": 100,  # Minimum viable chunk size
    "MAX_FILE_SIZE_MB": 50,  # Maximum file size limit
}

# Enhanced Helper Functions
def create_file_path(filename: str) -> str:
    """Ensure DB path exists and create a safe file path with validation"""
    try:
        os.makedirs(CONFIG["DB_PATH"], exist_ok=True)
        safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
        return os.path.join(CONFIG["DB_PATH"], safe_filename)
    except Exception as e:
        logger.error(f"Error creating file path: {e}")
        raise

def validate_file(uploaded_file) -> Tuple[bool, str]:
    """Validate uploaded file before processing"""
    if not uploaded_file:
        return False, "No file provided"
    
    if not uploaded_file.name.lower().endswith('.pdf'):
        return False, "Only PDF files are supported"
    
    file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
    if file_size_mb > CONFIG["MAX_FILE_SIZE_MB"]:
        return False, f"File too large ({file_size_mb:.1f}MB). Maximum size: {CONFIG['MAX_FILE_SIZE_MB']}MB"
    
    return True, "Valid file"

def update_conversation_memory(question: str, answer: str, sources: set):
    """Maintain a sliding window of recent conversations for context"""
    memory_entry = {
        'question': question,
        'answer': answer,
        'sources': list(sources),
        'timestamp': datetime.now().isoformat()
    }
    
    # Add to memory
    st.session_state.conversation_memory.append(memory_entry)
    
    # Keep only last N conversations
    if len(st.session_state.conversation_memory) > CONFIG["MEMORY_SIZE"]:
        st.session_state.conversation_memory = st.session_state.conversation_memory[-CONFIG["MEMORY_SIZE"]:]

def get_conversation_context() -> str:
    """Generate context from recent conversation history"""
    if not st.session_state.conversation_memory:
        return ""
    
    context_parts = ["RECENT CONVERSATION HISTORY:"]
    for i, entry in enumerate(st.session_state.conversation_memory[-3:], 1):  # Use last 3 for context
        context_parts.append(f"\nQ{i}: {entry['question'][:200]}...")
        context_parts.append(f"A{i}: {entry['answer'][:300]}...")
    
    context_parts.append("\nCURRENT QUESTION:")
    return "\n".join(context_parts)

def wrap_text(text: str, width: int = 90) -> str:
    """Wrap long text for better readability with error handling"""
    if not text or not isinstance(text, str):
        return ""
    try:
        return "\n".join([text[i:i+width] for i in range(0, len(text), width)])
    except Exception:
        return text

# Enhanced Document Processing with retry mechanism
def safe_extract_text(pdf_path: str) -> str:
    """Robust text extraction with multiple fallback mechanisms and retries"""
    for attempt in range(CONFIG["MAX_RETRIES"]):
        try:
            # Method 1: PyPDF (fastest)
            try:
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                text = " ".join([page.page_content for page in pages])
                if text.strip() and len(text) > CONFIG["MIN_CHUNK_LENGTH"]:
                    logger.info(f"Successfully extracted {len(text)} characters using PyPDF")
                    return text
            except Exception as e:
                logger.warning(f"PyPDF extraction failed (attempt {attempt+1}): {e}")
            
            # Method 2: Unstructured (more robust)
            try:
                loader = UnstructuredPDFLoader(pdf_path)
                data = loader.load()
                if data and data[0].page_content:
                    text = data[0].page_content
                    if len(text) > CONFIG["MIN_CHUNK_LENGTH"]:
                        logger.info(f"Successfully extracted {len(text)} characters using Unstructured")
                        return text
            except Exception as e:
                logger.warning(f"Unstructured extraction failed (attempt {attempt+1}): {e}")
            
            if attempt < CONFIG["MAX_RETRIES"] - 1:
                time.sleep(1)  # Brief pause before retry
                
        except Exception as e:
            logger.error(f"Critical error in text extraction (attempt {attempt+1}): {e}")
    
    return ""

def preprocess_text(text: str) -> str:
    """Advanced text preprocessing for better chunking"""
    if not text:
        return ""
    
    # Remove excessive whitespace while preserving structure
    text = re.sub(r'\n{3,}', '\n\n', text)  # Limit consecutive newlines
    text = re.sub(r'[ \t]{2,}', ' ', text)   # Remove excessive spaces/tabs
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\'\/\%\$\@\#\&\*\+\=\<\>\~\`]', ' ', text)  # Clean special chars
    
    # Preserve important formatting
    text = re.sub(r'(\d+\.)\s*([A-Z])', r'\1 \2', text)  # Fix numbered lists
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)     # Add space between camelCase
    
    return text.strip()

def process_pdf_chunk(pdf_path: str, vector_db, filename: str) -> int:
    """Enhanced PDF processing with better error handling and chunking strategy"""
    try:
        # Extract and preprocess text
        raw_text = safe_extract_text(pdf_path)
        if not raw_text.strip():
            st.error(f"⛔ Failed to extract meaningful text from: {filename}")
            return 0
        
        # Preprocess the text
        text = preprocess_text(raw_text)
        
        # Limit text size to prevent memory issues
        max_chars = 800000  # Increased limit for better coverage
        if len(text) > max_chars:
            text = text[:max_chars]
            logger.warning(f"Text truncated to {max_chars} characters for {filename}")
        
        # Smart chunking with multiple strategies
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["CHUNK_SIZE"],
            chunk_overlap=CONFIG["CHUNK_OVERLAP"],
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Better separation strategy
        )
        
        chunks = splitter.split_text(text)
        
        # Filter out very small chunks
        valid_chunks = [chunk for chunk in chunks if len(chunk.strip()) >= CONFIG["MIN_CHUNK_LENGTH"]]
        
        if not valid_chunks:
            st.error(f"⛔ No valid chunks generated from: {filename}")
            return 0
        
        # Create enhanced metadata
        metadatas = []
        for i, chunk in enumerate(valid_chunks):
            metadata = {
                "source": filename,
                "chunk_id": i,
                "chunk_length": len(chunk),
                "processing_date": datetime.now().isoformat()
            }
            metadatas.append(metadata)
        
        # Add to vector store with error handling
        try:
            vector_db.add_texts(texts=valid_chunks, metadatas=metadatas)
            logger.info(f"Successfully added {len(valid_chunks)} chunks from {filename}")
            return len(valid_chunks)
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            return 0
            
    except Exception as e:
        logger.error(f"Error processing {filename}: {traceback.format_exc()}")
        st.error(f"❌ Processing error for {filename}: {str(e)}")
        return 0

# Enhanced Knowledge Base Functions
def build_knowledge_base(uploaded_files=[]):
    """Enhanced knowledge base building with better error handling and progress tracking"""
    if not uploaded_files:
        st.warning("No files provided for knowledge base building")
        return 0
    
    try:
        # Validate all files first
        valid_files = []
        for file in uploaded_files:
            is_valid, message = validate_file(file)
            if is_valid:
                valid_files.append(file)
            else:
                st.error(f"❌ {file.name}: {message}")
        
        if not valid_files:
            st.error("No valid files to process")
            return 0
        
        # Initialize embeddings with error handling
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=API_KEY,
                task_type="retrieval_document"
            )
        except Exception as e:
            st.error(f"❌ Failed to initialize embeddings: {e}")
            return 0
        
        # Initialize or load Chroma vector store
        try:
            vector_db = Chroma(
                persist_directory=CONFIG["DB_PATH"],
                embedding_function=embeddings,
                collection_name="cfa_knowledge"
            )
        except:
            try:
                # Create new collection if it doesn't exist
                vector_db = Chroma.from_texts(
                    texts=["Initialize collection"],
                    embedding=embeddings,
                    persist_directory=CONFIG["DB_PATH"],
                    collection_name="cfa_knowledge"
                )
            except Exception as e:
                st.error(f"❌ Failed to initialize vector database: {e}")
                return 0
        
        total_chunks = 0
        source_files = set()
        failed_files = []
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each valid file
        for i, uploaded_file in enumerate(valid_files):
            try:
                file_path = create_file_path(uploaded_file.name)
                
                # Save uploaded file to disk
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process the file
                start_time = time.time()
                chunks_added = process_pdf_chunk(file_path, vector_db, uploaded_file.name)
                processing_time = time.time() - start_time
                
                if chunks_added > 0:
                    total_chunks += chunks_added
                    source_files.add(uploaded_file.name)
                    if uploaded_file.name not in st.session_state.uploaded_files:
                        st.session_state.uploaded_files.append(uploaded_file.name)
                    
                    status_msg = f"✅ {uploaded_file.name}: {chunks_added} chunks ({processing_time:.1f}s)"
                    logger.info(status_msg)
                else:
                    failed_files.append(uploaded_file.name)
                    status_msg = f"❌ {uploaded_file.name}: Processing failed"
                
                # Update progress
                progress = (i + 1) / len(valid_files)
                progress_bar.progress(progress)
                status_text.text(f"📖 Processing {i+1}/{len(valid_files)}: {status_msg}")
                
                # Memory management
                gc.collect()
                time.sleep(0.1)  # Brief pause to prevent overwhelming
                
            except Exception as e:
                failed_files.append(uploaded_file.name)
                logger.error(f"Failed to process {uploaded_file.name}: {e}")
                continue
        
        # Finalize
        try:
            st.session_state.db_exists = True
            st.session_state.sources = source_files
            logger.info(f"Knowledge base updated with {total_chunks} total chunks")
        except Exception as e:
            st.error(f"❌ Failed to update session state: {e}")
            return 0
        
        # Clear progress indicators
        progress_bar.empty()
        
        # Display final status
        if total_chunks > 0:
            success_msg = f"✅ Knowledge base built successfully!\n- {total_chunks} chunks from {len(source_files)} files"
            if failed_files:
                success_msg += f"\n- ⚠️ {len(failed_files)} files failed: {', '.join(failed_files[:3])}{'...' if len(failed_files) > 3 else ''}"
            status_text.success(success_msg)
        else:
            status_text.error("❌ No content was successfully added to the knowledge base")
            
        return total_chunks
    
    except Exception as e:
        logger.error(f"Critical error in build_knowledge_base: {traceback.format_exc()}")
        st.error(f"❌ Critical system error: {str(e)}")
        return 0

# Enhanced Reasoning Engine with improved prompts
def modern_reasoning_engine(question: str, context: str, conversation_history: str = "") -> str:
    """Advanced reasoning pipeline with enhanced prompts and error handling"""
    
    # Enhanced system prompt with better CFA-specific instructions
    SYSTEM_PROMPT = """You are an expert CFA (Chartered Financial Analyst) instructor with deep knowledge across all three levels of the CFA curriculum. Your role is to provide comprehensive, accurate, and educational responses that help candidates understand complex financial concepts.

INSTRUCTIONS:
1. **Analyze the question** to determine the CFA level (I, II, III) and topic area
2. **Use ONLY the provided context** - do not add external information
3. **Structure your response** with clear sections when appropriate
4. **Include relevant formulas** and step-by-step calculations
5. **Provide practical examples** when they enhance understanding
6. **Reference specific CFA concepts** and reading assignments when mentioned in context

RESPONSE STRUCTURE:
- **Concept Identification**: CFA level and topic area
- **Key Definitions**: Define important terms
- **Detailed Explanation**: Step-by-step breakdown
- **Formulas/Calculations**: When applicable
- **Practical Application**: Real-world relevance
- **Key Takeaways**: Summary points

{conversation_context}

CONTEXT FROM CFA MATERIALS:
{context}

QUESTION: {question}

Provide a comprehensive answer based strictly on the context provided above. If the context doesn't contain sufficient information, clearly state what aspects cannot be answered from the available materials."""

    try:
        # Initialize model with enhanced configuration
        llm = ChatGoogleGenerativeAI(
            model=st.session_state.model_name,
            google_api_key=API_KEY,
            temperature=CONFIG["LLM_TEMPERATURE"],
            max_output_tokens=CONFIG["LLM_MAX_TOKENS"],
            top_p=0.95,  # Added for more focused responses
            top_k=40     # Added for better token selection
        )
        
        # Prepare context with size management
        safe_context = context[:25000]  # Increased context window
        safe_history = conversation_history[:5000] if conversation_history else ""
        safe_question = question[:1000]  # Reasonable question length limit
        
        # Format the complete prompt
        formatted_prompt = SYSTEM_PROMPT.format(
            conversation_context=f"CONVERSATION HISTORY:\n{safe_history}\n" if safe_history else "",
            context=safe_context,
            question=safe_question
        )
        
        # Generate response with timeout handling
        start_time = time.time()
        response = llm.invoke(formatted_prompt)
        generation_time = time.time() - start_time
        
        if hasattr(response, 'content'):
            answer = response.content
        else:
            answer = str(response)
        
        # Validate and clean response
        if not answer or len(answer.strip()) < 50:
            return "⚠️ The generated response was too brief. Please rephrase your question or provide more context."
        
        # Limit response length and add generation info
        final_answer = answer[:CONFIG["MAX_ANSWER_LENGTH"]]
        if len(answer) > CONFIG["MAX_ANSWER_LENGTH"]:
            final_answer += "\n\n*[Response truncated for length]*"
            
        logger.info(f"Generated response in {generation_time:.2f}s, length: {len(final_answer)}")
        return final_answer
        
    except Exception as e:
        error_msg = f"Answer generation failed: {str(e)}"
        logger.error(f"Error in modern_reasoning_engine: {traceback.format_exc()}")
        
        # Update error tracking
        st.session_state.error_count += 1
        
        return f"⚠️ {error_msg}. Please try rephrasing your question or check your internet connection."

# Enhanced Query Handling with retry mechanism
def retrieve_context(question: str) -> Tuple[str, set]:
    """Enhanced context retrieval with better error handling and retry logic"""
    for attempt in range(CONFIG["MAX_RETRIES"]):
        try:
            # Initialize embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=API_KEY,
                task_type="retrieval_query"  # Optimized for query tasks
            )
            
            # Load vector database
            vector_db = Chroma(
                persist_directory=CONFIG["DB_PATH"],
                embedding_function=embeddings,
                collection_name="cfa_knowledge"
            )
            
            # Enhanced retrieval with multiple strategies
            docs = vector_db.similarity_search(
                question, 
                k=CONFIG["RAG_TOPK"],
                filter=None  # Could add metadata filtering here
            )
            
            if not docs:
                logger.warning(f"No documents retrieved for question: {question[:100]}")
                return "No relevant information found in the knowledge base.", set()
            
            # Build enhanced context
            context_parts = []
            sources = set()
            
            for i, doc in enumerate(docs):
                source = doc.metadata.get("source", "Unknown Source")
                chunk_id = doc.metadata.get("chunk_id", "unknown")
                sources.add(source)
                
                # Enhanced context formatting
                context_section = f"""
DOCUMENT SECTION {i+1}:
Source: {source} (Chunk {chunk_id})
Content: {doc.page_content[:1000]}
---"""
                context_parts.append(context_section)
            
            context = "\n".join(context_parts)
            logger.info(f"Retrieved {len(docs)} documents from {len(sources)} sources")
            
            return context, sources
            
        except Exception as e:
            logger.error(f"Error in retrieve_context (attempt {attempt+1}): {e}")
            if attempt < CONFIG["MAX_RETRIES"] - 1:
                time.sleep(2)  # Wait before retry
                continue
            else:
                return f"⚠️ Context retrieval failed after {CONFIG['MAX_RETRIES']} attempts", set()

def process_query(question: str) -> Tuple[str, set, float]:
    """Enhanced end-to-end query processing with comprehensive error handling"""
    start_time = time.time()
    
    try:
        # Update query statistics
        st.session_state.total_queries += 1
        
        # Validate question
        if not question or len(question.strip()) < 5:
            return "⚠️ Please provide a more detailed question about CFA topics.", set(), 0
        
        # Check if knowledge base exists
        if not st.session_state.db_exists:
            return "⚠️ Please build the knowledge base first by uploading CFA materials.", set(), 0
        
        # Retrieve context
        context, sources = retrieve_context(question)
        
        if not context or "No relevant information found" in context:
            suggestion = "Try rephrasing your question or ensure your uploaded materials cover this topic."
            return f"⚠️ No relevant information found in knowledge base. {suggestion}", set(), time.time() - start_time
        
        # Get conversation history for context
        conversation_context = get_conversation_context()
        
        # Generate reasoned response
        answer = modern_reasoning_engine(question, context, conversation_context)
        
        processing_time = time.time() - start_time
        
        # Update conversation memory
        update_conversation_memory(question, answer, sources)
        
        # Update performance metrics
        current_avg = st.session_state.average_response_time
        total_queries = st.session_state.total_queries
        st.session_state.average_response_time = ((current_avg * (total_queries - 1)) + processing_time) / total_queries
        
        logger.info(f"Query processed successfully in {processing_time:.2f}s")
        return answer, sources, processing_time
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Query processing failed: {str(e)}"
        logger.error(f"Error in process_query: {traceback.format_exc()}")
        
        st.session_state.error_count += 1
        
        return f"⚠️ {error_msg}", set(), processing_time

# Enhanced UI Components
def render_message(role: str, content: str, sources: set = set()):
    """Enhanced message rendering with better formatting and source management"""
    avatar = "💼" if role == "assistant" else "🧑‍🎓"
    
    with st.chat_message(role, avatar=avatar):
        # Render main content
        st.markdown(content)
        
        # Enhanced source display
        if sources and role == "assistant":
            with st.expander(f"📚 Source Documents ({len(sources)} files)", expanded=False):
                for i, source in enumerate(sorted(sources), 1):
                    st.write(f"{i}. **{source}**")

def render_system_status():
    """Display system health and performance metrics"""
    with st.sidebar:
        st.markdown("---")
        st.subheader("📊 System Status")
        
        # Health indicator
        health_color = "🟢" if st.session_state.error_count < 3 else "🟡" if st.session_state.error_count < 10 else "🔴"
        st.write(f"{health_color} System Health")
        
        # Performance metrics
        if st.session_state.total_queries > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Queries", st.session_state.total_queries)
            with col2:
                st.metric("Avg Response", f"{st.session_state.average_response_time:.1f}s")
            
            if st.session_state.error_count > 0:
                st.metric("Error Rate", f"{(st.session_state.error_count/st.session_state.total_queries)*100:.1f}%")

# Main Streamlit Application
def main():
    # Page setup with enhanced configuration
    st.set_page_config(
        page_title="CFA Expert Assistant - Enhanced",
        page_icon="💼",
        layout="centered",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo/cfa-assistant',
            'Report a bug': 'https://github.com/your-repo/cfa-assistant/issues',
            'About': "CFA Assistant v3.0 - Advanced RAG System with Memory"
        }
    )

    # Enhanced CSS styling
    st.markdown("""
    <style>
        .stApp { 
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .main-header {
            text-align: center;
            padding: 1rem 0;
            background: linear-gradient(90deg, #1e3a66, #2d5aa0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }
        .chat-container {
            background: white;
            border-radius: 15px;
            padding: 1rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stChatMessage {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e3a66 0%, #2d5aa0 100%);
            color: white;
        }
        .sidebar-title {
            color: white !important;
            font-weight: bold;
        }
        .metric-container {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 0.5rem;
            margin: 0.25rem 0;
        }
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #1e88e5, #42a5f5) !important;
        }
        .success-banner {
            background: linear-gradient(90deg, #4caf50, #66bb6a);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    init_session_state()

    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("<h1 class='sidebar-title'>🎓 CFA Knowledge Hub</h1>", unsafe_allow_html=True)
        
        # About section
        with st.expander("🎯 Enhanced Features", expanded=False):
            st.info("""
            **New in v3.0:**
            - 🧠 **Conversation Memory** (last 5 Q&A)
            - 🔄 **Auto-retry** mechanisms
            - 📊 **Performance tracking**
            - ✅ **Enhanced validation**
            - 🎯 **Better CFA-specific prompts**
            - 🔍 **Improved context retrieval**
            """)
        
        st.markdown("---")
        
        # File upload section
        st.subheader("📚 Knowledge Base Management")
        
        uploaded_files = st.file_uploader(
            "Upload CFA PDF Materials",
            type="pdf",
            accept_multiple_files=True,
            help=f"Max file size: {CONFIG['MAX_FILE_SIZE_MB']}MB. Supports: Curriculum, Schweser, Mock Exams"
        )
        
        # Build knowledge base button
        if st.button("🏗️ Build/Update Knowledge Base", use_container_width=True, type="primary"):
            if uploaded_files:
                with st.spinner("🔄 Processing documents... This may take several minutes."):
                    chunks_added = build_knowledge_base(uploaded_files)
                    if chunks_added > 0:
                        st.balloons()  # Celebrate success
            else:
                st.warning("⚠️ Please upload PDF files first")
        
        # Current materials display
        if st.session_state.uploaded_files:
            st.markdown("---")
            st.subheader("📖 Current Materials")
            
            # Show files with better formatting
            for i, file in enumerate(st.session_state.uploaded_files[:8], 1):
                st.text(f"{i}. {file}")
            
            if len(st.session_state.uploaded_files) > 8:
                st.caption(f"... and {len(st.session_state.uploaded_files)-8} more files")
            
            # Clear knowledge base option
            if st.button("🗑️ Clear Knowledge Base", help="Remove all uploaded materials"):
                if st.checkbox("I understand this will delete all materials"):
                    st.session_state.uploaded_files = []
                    st.session_state.db_exists = False
                    st.session_state.sources = set()
                    st.success("Knowledge base cleared!")
                    st.rerun()
        
        # System status
        render_system_status()
        
        st.markdown("---")
        st.caption("💡 **Pro Tips:**")
        st.caption("• Ask specific questions about formulas, concepts, or calculations")
        st.caption("• Reference specific CFA readings or topics")
        st.caption("• The system remembers your last 5 questions for context")

    # Main content area
    st.markdown("<h1 class='main-header'>CFA Expert Assistant v3.0</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Enhanced with Memory & Advanced Reasoning • Powered by Gemini AI</p>", unsafe_allow_html=True)

    # Welcome message for new users
    if st.session_state.first_load:
        st.markdown("""
        <div class='success-banner'>
            <h3>🎉 Welcome to Your Enhanced CFA Study Assistant!</h3>
            <p>Get started in 3 easy steps:</p>
            <p>1️⃣ Upload your CFA materials (PDFs) → 2️⃣ Build knowledge base → 3️⃣ Ask questions!</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.chat_message("assistant", avatar="💼"):
            st.markdown("""
            Hello! I'm your enhanced CFA study assistant with several new capabilities:
            
            🧠 **Memory**: I remember our last 5 conversations for better context
            🎯 **Smart Reasoning**: Enhanced prompts for detailed CFA-specific explanations  
            🔄 **Reliability**: Auto-retry mechanisms and better error handling
            📊 **Tracking**: Performance monitoring and system health indicators
            
            Upload your CFA materials and let's start studying! 📚
            """)
        st.session_state.first_load = False

    # Display conversation history with enhanced formatting
    for message in st.session_state.conversation:
        render_message(**message)

    # Enhanced chat input with better UX
    if st.session_state.db_exists:
        placeholder_text = "Ask about CFA concepts, formulas, or specific topics..."
        if st.session_state.conversation_memory:
            placeholder_text += " (I remember our recent conversation)"
    else:
        placeholder_text = "Please build your knowledge base first..."

    # Query processing with enhanced error handling
    query = st.chat_input(
        placeholder_text, 
        key="chat_input",
        disabled=not st.session_state.db_exists or st.session_state.processing
    )

    if query and not st.session_state.processing:
        # Add user message to conversation
        st.session_state.conversation.append({"role": "user", "content": query})
        render_message("user", query)

        # Process the query with enhanced UX
        st.session_state.processing = True
        
        with st.spinner("🔍 Analyzing your question and searching CFA materials..."):
            try:
                # Create a placeholder for streaming-like effect
                with st.chat_message("assistant", avatar="💼"):
                    thinking_placeholder = st.empty()
                    thinking_placeholder.markdown("🤔 Thinking...")
                    
                    # Process the query
                    answer, sources, processing_time = process_query(query)
                    
                    # Clear thinking indicator and show response
                    thinking_placeholder.empty()
                    st.markdown(answer)
                    
                    # Add to conversation history
                    st.session_state.conversation.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                    # Enhanced source display
                    if sources:
                        with st.expander(f"📚 Source Documents ({len(sources)} files)", expanded=False):
                            for i, source in enumerate(sorted(sources), 1):
                                st.write(f"{i}. **{source}**")
                    
                    # Performance info with better formatting
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"⏱️ {processing_time:.1f}s")
                    with col2:
                        if sources:
                            st.caption(f"📄 {len(sources)} sources")
                    with col3:
                        st.caption(f"🧠 Memory: {len(st.session_state.conversation_memory)}/5")
                    
            except Exception as e:
                st.error(f"❌ System error: {str(e)}")
                logger.error(f"Critical error in main query processing: {traceback.format_exc()}")
                
                # Add error message to conversation
                st.session_state.conversation.append({
                    "role": "assistant",
                    "content": "⚠️ I encountered a technical issue. Please try again or rephrase your question.",
                    "sources": set()
                })
        
        st.session_state.processing = False
        
        # Trigger garbage collection to manage memory
        gc.collect()

    # Footer with additional information
    st.markdown("---")
    
    # Performance dashboard (collapsible)
    with st.expander("📊 Session Statistics", expanded=False):
        if st.session_state.total_queries > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Questions", 
                    st.session_state.total_queries,
                    help="Number of questions asked this session"
                )
            
            with col2:
                st.metric(
                    "Average Response Time", 
                    f"{st.session_state.average_response_time:.1f}s",
                    help="Average time to generate responses"
                )
            
            with col3:
                success_rate = ((st.session_state.total_queries - st.session_state.error_count) 
                               / st.session_state.total_queries * 100) if st.session_state.total_queries > 0 else 100
                st.metric(
                    "Success Rate", 
                    f"{success_rate:.1f}%",
                    help="Percentage of successful responses"
                )
            
            with col4:
                st.metric(
                    "Memory Usage", 
                    f"{len(st.session_state.conversation_memory)}/5",
                    help="Conversation history maintained for context"
                )
            
            # Memory content preview
            if st.session_state.conversation_memory:
                st.subheader("🧠 Recent Conversation Memory")
                for i, memory in enumerate(st.session_state.conversation_memory[-3:], 1):
                    st.text(f"Q{i}: {memory['question'][:100]}...")
                    st.text(f"A{i}: {memory['answer'][:150]}...")
                    st.text("---")
        else:
            st.info("Ask your first question to see session statistics!")
    
    # Help section
    with st.expander("❓ How to Get the Best Results", expanded=False):
        st.markdown("""
        ### 🎯 Question Types That Work Best:
        - **Conceptual**: "Explain the difference between forward and futures contracts"
        - **Calculation**: "How do I calculate the Sharpe ratio and what does it mean?"
        - **Application**: "When would I use CAPM vs. APT in portfolio management?"
        - **Comparison**: "Compare DCF and relative valuation methods"
        
        ### 💡 Tips for Better Responses:
        - Be specific about the CFA level (I, II, III) if relevant
        - Mention specific readings or study sessions when possible
        - Ask follow-up questions to dive deeper into topics
        - Reference formulas or calculations you're struggling with
        
        ### 🔧 Troubleshooting:
        - If responses seem incomplete, try breaking complex questions into parts
        - For calculation problems, specify what inputs you have
        - If sources seem irrelevant, try rephrasing with more specific terminology
        """)
    
    # Model selection (advanced users)
    with st.expander("⚙️ Advanced Settings", expanded=False):
        st.subheader("Model Configuration")
        
        model_options = {
            "gemini-1.5-flash": "Fast & Efficient (Recommended)",
            "gemini-1.5-pro": "Enhanced Reasoning (Slower)",
        }
        
        selected_model = st.selectbox(
            "Choose AI Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0 if st.session_state.model_name == "gemini-1.5-flash" else 1,
            help="Flash model is faster, Pro model provides more detailed analysis"
        )
        
        if selected_model != st.session_state.model_name:
            st.session_state.model_name = selected_model
            st.success(f"Model changed to {model_options[selected_model]}")
        
        # Advanced configuration options
        st.subheader("Retrieval Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            new_topk = st.slider(
                "Context Documents", 
                min_value=3, 
                max_value=10, 
                value=CONFIG["RAG_TOPK"],
                help="Number of document chunks to retrieve for context"
            )
        
        with col2:
            new_temp = st.slider(
                "Response Creativity", 
                min_value=0.0, 
                max_value=1.0, 
                value=CONFIG["LLM_TEMPERATURE"],
                step=0.1,
                help="Higher values = more creative, lower = more focused"
            )
        
        if st.button("Apply Settings"):
            CONFIG["RAG_TOPK"] = new_topk
            CONFIG["LLM_TEMPERATURE"] = new_temp
            st.success("Settings updated!")
    
    # Memory management controls
    if st.session_state.conversation_memory:
        with st.expander("🧠 Memory Management", expanded=False):
            st.write(f"Current memory: {len(st.session_state.conversation_memory)} conversations")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear Memory"):
                    st.session_state.conversation_memory = []
                    st.success("Conversation memory cleared!")
                    st.rerun()
            
            with col2:
                if st.button("Export Memory"):
                    memory_json = json.dumps(st.session_state.conversation_memory, indent=2)
                    st.download_button(
                        "Download Memory",
                        memory_json,
                        file_name=f"cfa_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

# Error handling and recovery
def handle_critical_error():
    """Handle critical system errors gracefully"""
    st.error("🚨 Critical system error detected. Attempting recovery...")
    
    # Reset critical session state
    st.session_state.processing = False
    st.session_state.error_count += 1
    
    # Attempt to clear problematic state
    if st.session_state.error_count > 10:
        st.warning("Multiple errors detected. Consider refreshing the page.")
        if st.button("Reset Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# Application health check
def perform_health_check():
    """Perform basic system health check"""
    try:
        # Check API connectivity
        test_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=API_KEY
        )
        
        # Check database accessibility
        if st.session_state.db_exists:
            vector_db = Chroma(
                persist_directory=CONFIG["DB_PATH"],
                embedding_function=test_embeddings,
                collection_name="cfa_knowledge"
            )
        
        st.session_state.system_health['status'] = 'healthy'
        st.session_state.system_health['last_check'] = datetime.now()
        
    except Exception as e:
        st.session_state.system_health['status'] = 'unhealthy'
        st.session_state.system_health['last_error'] = str(e)
        logger.error(f"Health check failed: {e}")

# Run the application
if __name__ == "__main__":
    try:
        main()
        
        # Periodic health check (every 10 queries)
        if st.session_state.total_queries % 10 == 0 and st.session_state.total_queries > 0:
            perform_health_check()
            
    except Exception as e:
        logger.critical(f"Application crashed: {traceback.format_exc()}")
        handle_critical_error()