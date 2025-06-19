import os
import re
import gc
import time
import json
import shutil
import logging
import hashlib
import traceback
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import streamlit as st

# Import with proper error handling
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError as e:
    st.error(f"Missing required package: {e}")
    st.info("Please install required packages:\npip install langchain-community langchain-google-genai langchain-chroma chromadb protobuf==3.20.0 pypdf2 pdfplumber")
    st.stop()

# Configure robust logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cfa_assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CFA_Assistant_Pro")

# Set page config FIRST - required by Streamlit
st.set_page_config(
    page_title="CFA Expert Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "CFA Assistant v12.0 • Robust Document Processing"
    }
)

# System constants
MAX_FILE_SIZE_MB = 102400  # 100 GB max
EMBEDDING_MODEL = "models/embedding-001"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
BATCH_SIZE = 10  # Reduced batch size for better stability
MAX_TEXT_LENGTH = 500000  # Character limit per document

# Ensure temp directories exist
os.makedirs("./vector_db", exist_ok=True)
os.makedirs("./processing", exist_ok=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    default_state = {
        'conversation': [],
        'processing': False,
        'kb_processing': False,
        'db_exists': False,
        'uploaded_files': set(),
        'processing_files': set(),
        'progress': 0.0,
        'progress_stage': "",
        'files_processed': 0,
        'files_failed': 0,
        'total_chunks': 0,
        'model': "gemini-1.5-flash",
        'file_status': {},
        'file_errors': {},
        'first_run': True,
        'error_message': "",
        'vector_db': None  # Cache the vector DB instance
    }
    
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Clear orphaned processing files
    if st.session_state['processing_files']:
        try:
            shutil.rmtree("./processing", ignore_errors=True)
            st.session_state['processing_files'] = set()
        except:
            pass

def check_db_exists():
    """Check if vector database exists"""
    try:
        db_path = Path("./vector_db")
        if db_path.exists() and db_path.is_dir():
            # Check if directory contains ChromaDB files
            required_files = ['chroma.sqlite3']
            return all(db_path.joinpath(f).exists() for f in required_files)
        return False
    except Exception as e:
        logger.error(f"Error checking DB existence: {e}")
        return False

# Initialize session state at startup
init_session_state()

# Set state of database existence
st.session_state['db_exists'] = check_db_exists()

# Load environment variables with validation
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("🔑 GEMINI_API_KEY not found in environment variables")
    logger.critical("Missing GEMINI_API_KEY in environment")
    st.stop()

# Custom CSS for robust UI
st.markdown("""
<style>
    :root {
        --background: #0f1e2d;
        --surface: #1d2e42;
        --primary: #1a73e8;
        --secondary: #57a6fa;
        --success: #2ecc71;
        --warning: #f1c40f;
        --error: #e74c3c;
        --text-primary: #ecf0f1;
        --text-secondary: #bdc3c7;
        --border: #3498db;
    }
    
    body, [data-testid="stAppViewContainer"] {
        background-color: var(--background) !important;
        color: var(--text-primary) !important;
        background-image: radial-gradient(circle at top right, rgba(41, 128, 185, 0.1), transparent 70%);
    }
    
    .header {
        background: linear-gradient(135deg, var(--primary), #0d47a1);
        padding: 1.8rem;
        border-radius: 0 0 30px 30px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
    }
    
    .header-title {
        color: white;
        font-size: 2.7rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.85);
        font-size: 1.25rem;
        font-weight: 300;
    }
    
    .card {
        background: rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    
    .status-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 24px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    .status-processing {
        background: linear-gradient(135deg, var(--warning), #f39c12);
        animation: pulse 1.5s infinite;
    }
    
    .status-completed {
        background: linear-gradient(135deg, var(--success), #27ae60);
    }
    
    .status-failed {
        background: linear-gradient(135deg, var(--error), #c0392b);
    }
    
    .error-box {
        background: rgba(231, 76, 60, 0.15);
        border: 1px solid var(--error);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
    
    .step-box {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .step-number {
        background: var(--primary);
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 15px;
        font-weight: bold;
    }
    
    .metric-display {
        display: flex;
        align-items: center;
        padding: 0.8rem;
        background: rgba(26, 115, 232, 0.15);
        border-radius: 15px;
        margin: 0.3rem 0;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-left: auto;
        min-width: 80px;
    }
    
    button {
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
init_session_state()

# Helper functions for session state access
def get_state(key, default=None):
    """Safely get a value from session state"""
    return st.session_state.get(key, default)

def set_state(key, value):
    """Safely set a value in session state"""
    st.session_state[key] = value

def update_state(updates):
    """Update multiple session state values at once"""
    for key, value in updates.items():
        st.session_state[key] = value

# Helper functions with robust error handling
def get_file_checksum(file_path):
    """Generate MD5 checksum for file integrity"""
    try:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Error getting checksum for {file_path}: {e}")
        return None

def preprocess_text(text):
    """Memory-efficient text processing"""
    # Remove excessive whitespace and control characters
    text = re.sub(r'\s+', ' ', text).strip()
    # Basic cleanup: remove characters not typically found in document text
    # This might need refinement based on actual document content
    # text = re.sub(r'[^\w\s.,;:\-!?]', ' ', text) # too aggressive, keep more characters
    return text[:MAX_TEXT_LENGTH]  # Truncate to avoid memory overload

def process_single_pdf(file_path, filename):
    """Process a single PDF efficiently, returning chunks and any error message."""
    chunks = []
    error_msg = ""
    try:
        logger.info(f"Starting PDF processing for: {filename}")
        loader = PyPDFLoader(file_path)
        
        raw_text = ""
        # Load all pages and extract text
        pages = loader.load()
        for page in pages:
            if page.page_content:  # Check if page has content
                raw_text += page.page_content + "\n"

        if not raw_text.strip():
            raise ValueError(f"No text extracted from PDF: {filename}")
        
        text = preprocess_text(raw_text)
        
        # Split into chunks with proper error handling
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks_list = splitter.split_text(text)
        
        if not chunks_list:
            raise ValueError(f"Text splitter produced no chunks from {filename}")
        
        # Filter out very small chunks and format
        valid_chunks = [{"text": chunk.strip(), "source": filename} for chunk in chunks_list if len(chunk.strip()) > 50]
        
        if not valid_chunks:
            raise ValueError(f"No substantial chunks created from {filename}")
        
        logger.info(f"Created {len(valid_chunks)} chunks from {filename}")
        return valid_chunks, ""
        
    except Exception as e:
        error_msg = f"Processing failed for {filename}: {str(e)}"
        logger.error(f"Error processing {filename}: {traceback.format_exc()}")
        return [], error_msg

def get_vector_db():
    """Create or connect to vector database with detailed error handling"""
    try:
        # Return cached instance if available
        if get_state('vector_db') is not None:
            logger.info("Using cached vector database instance")
            return get_state('vector_db')
            
        logger.info("Initializing new vector database connection")
        
        # Test API key first with better error handling
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL,
                google_api_key=API_KEY,
                task_type="retrieval_document"
            )
            logger.info("Embeddings model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise ValueError(f"Embeddings initialization failed: {str(e)}")
        
        # Initialize Chroma with proper settings and error handling
        try:
            db_path = "./vector_db"
            os.makedirs(db_path, exist_ok=True)
            
            # Import Chroma settings for better compatibility
            from chromadb.config import Settings
            
            # Initialize with specific settings for stability
            vector_db = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings,
                collection_name="cfa_knowledge",
                client_settings=Settings(
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
            
            # Test the connection
            try:
                vector_db.similarity_search("test", k=1)
                logger.info("Vector database test search successful")
            except Exception as test_e:
                logger.warning(f"Initial test search failed (this is normal for new DB): {test_e}")
            
            logger.info("Chroma vector database initialized successfully")
            
            # Cache the vector DB instance
            set_state('vector_db', vector_db)
            return vector_db
            
        except Exception as e:
            error_msg = f"Failed to initialize vector database: {str(e)}"
            logger.error(error_msg)
            set_state('error_message', error_msg)
            set_state('vector_db', None)
            return None
            
    except Exception as e:
        error_msg = f"VectorDB Error: {str(e)}"
        logger.error(error_msg)
        set_state('error_message', error_msg)
        set_state('vector_db', None)
        return None

def build_knowledge_base(uploaded_files):
    """Robust knowledge base builder with detailed progress tracking"""
    if not uploaded_files:
        set_state('error_message', "No files uploaded")
        return False
        
    try:
        # Initialize processing state
        update_state({
            'kb_processing': True,
            'progress': 0,
            'progress_stage': "Initializing",
            'files_processed': 0,
            'files_failed': 0,
            'total_chunks': 0,
            'file_status': {},
            'file_errors': {},
            'error_message': ""
        })
        
        # Clean processing directory
        processing_dir = "./processing"
        try:
            shutil.rmtree(processing_dir, ignore_errors=True)
            os.makedirs(processing_dir, exist_ok=True)
            logger.info(f"Processing directory prepared: {processing_dir}")
        except Exception as e:
            error_msg = f"Failed to prepare processing directory: {str(e)}"
            set_state('error_message', error_msg)
            logger.error(error_msg)
            return False
        
        # Initialize vector DB
        set_state('progress_stage', "Connecting to vector database")
        set_state('progress', 5)
        set_state('vector_db', None)  # Force fresh connection
        
        vector_db = get_vector_db()
        if not vector_db:
            set_state('error_message', "Failed to initialize vector database")
            return False
        
        logger.info("Vector database connection established")
        
        # Process each file
        total_files = len(uploaded_files)
        all_chunks_for_db = []
        
        for idx, uploaded_file in enumerate(uploaded_files):
            filename = uploaded_file.name
            
            try:
                logger.info(f"Processing file {idx + 1}/{total_files}: {filename}")
                
                # Update progress
                st.session_state['file_status'][filename] = "processing"
                set_state('progress_stage', f"Processing {filename}")
                set_state('progress', 10 + ((idx / total_files) * 60))
                
                # Save and process file
                file_path = os.path.join(processing_dir, filename)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                if not os.path.exists(file_path):
                    raise Exception("File failed to save")
                
                # Process the PDF
                chunks, error = process_single_pdf(file_path, filename)
                if error:
                    raise Exception(error)
                    
                if not chunks:
                    raise Exception("No text extracted from PDF")
                
                # Update progress
                st.session_state['total_chunks'] += len(chunks)
                all_chunks_for_db.extend(chunks)
                st.session_state['file_status'][filename] = "completed"
                st.session_state['files_processed'] += 1
                
                logger.info(f"Successfully processed {filename}: {len(chunks)} chunks")
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Failed to process {filename}: {error_msg}")
                st.session_state['file_status'][filename] = "failed"
                st.session_state['file_errors'][filename] = error_msg
                st.session_state['files_failed'] += 1
            
            finally:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {file_path}: {e}")
        
        # Add to vector database
        if all_chunks_for_db:
            try:
                set_state('progress_stage', f"Adding {len(all_chunks_for_db)} chunks to vector database")
                set_state('progress', 70)
                
                # Process in smaller batches
                batch_size = min(BATCH_SIZE, 20)
                for i in range(0, len(all_chunks_for_db), batch_size):
                    batch = all_chunks_for_db[i:i + batch_size]
                    texts = [chunk["text"] for chunk in batch]
                    metadatas = [{"source": chunk["source"], "chunk_id": idx + i} for idx, chunk in enumerate(batch)]
                    
                    try:
                        vector_db.add_texts(texts=texts, metadatas=metadatas)
                    except Exception as e:
                        logger.error(f"Failed to add batch {i//batch_size + 1}: {str(e)}")
                        # Continue with next batch
                
                vector_db.persist()
                set_state('db_exists', True)
                
                logger.info(f"Successfully added {len(all_chunks_for_db)} chunks to vector database")
                return True
                
            except Exception as e:
                error_msg = f"Failed to update vector database: {str(e)}"
                set_state('error_message', error_msg)
                logger.error(error_msg)
                return False
        else:
            set_state('error_message', "No valid chunks generated from files")
            return False
            
    except Exception as e:
        error_msg = f"Knowledge base build failed: {str(e)}"
        set_state('error_message', error_msg)
        logger.error(error_msg)
        return False
        
    finally:
        set_state('kb_processing', False)
        shutil.rmtree(processing_dir, ignore_errors=True)

# Query processing functions
def handle_user_query(question):
    """Robust query handling"""
    logger.info(f"Received query: {question}")
    st.session_state.processing = True
    
    try:
        # Initialize vector DB
        vector_db = get_vector_db()
        if not vector_db:
            return "⚠️ Knowledge base not available. Please build the knowledge base first.", set(), 0
        
        # Check if database has content
        try:
            doc_count = vector_db._collection.count()
            if doc_count == 0:
                return "⚠️ Knowledge base is empty. Please upload and process documents first.", set(), 0
            logger.info(f"Querying database with {doc_count} documents")
        except Exception as e:
            logger.warning(f"Could not get collection count for empty check: {e}")
            pass # Continue, as it might just be a new DB
        
        # Retrieve context
        context, sources = retrieve_context(vector_db, question)
        if not context or not sources:
            return "⚠️ No relevant information found in the knowledge base for your query.", set(), 0
        
        # Generate response
        answer = generate_response(question, context)
        
        return answer, sources, 0
        
    except Exception as e:
        logger.error(f"Query failed: {traceback.format_exc()}")
        return f"⚠️ System error during processing: {str(e)}", set(), 0
    finally:
        st.session_state.processing = False

def retrieve_context(vector_db, question):
    """Context retrieval with error handling"""
    try:
        # Perform similarity search
        docs = vector_db.similarity_search(
            question, 
            k=5,
            filter=None
        )
        
        if not docs:
            logger.warning("No documents found in similarity search")
            return "", set()
        
        context_parts = []
        sources = set()
        
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            sources.add(source)
            
            # Clean and truncate content
            content = doc.page_content.strip()
            if len(content) > 800:
                content = content[:800] + "..."
            
            context_parts.append(f"Source: {source}\n{content}\n")
        
        context = "\n".join(context_parts)
        logger.info(f"Retrieved context from {len(sources)} sources, {len(context)} characters")
        
        return context, sources
        
    except Exception as e:
        logger.error(f"Context retrieval failed: {str(e)}")
        return "", set()

def generate_response(question, context):
    """Response generation with Gemini"""
    try:
        model = ChatGoogleGenerativeAI(
            model=st.session_state.model,
            google_api_key=API_KEY,
            temperature=0.3,
            max_output_tokens=2000
        )
        
        prompt = f"""
        As a CFA expert, provide a comprehensive and authoritative answer using only the context provided below.
        
        Context from CFA materials:
        {context[:10000]}
        
        Question: {question}
        
        Instructions:
        1. Answer based strictly on the provided context
        2. If the context doesn't contain enough information, clearly state this
        3. Structure your response with clear headings when appropriate
        4. Include relevant formulas or calculations if mentioned in the context
        5. Reference the source materials when making specific claims
        
        Provide a detailed, professional response:
        """
        
        result = model.invoke(prompt)
        
        if not result or not result.content:
            return "⚠️ No response generated from the AI model"
        
        return result.content
        
    except Exception as e:
        logger.error(f"Response generation failed: {str(e)}")
        return f"⚠️ Could not generate response: {str(e)}"

# UI Components
def header_section():
    """Shared header"""
    st.markdown("""
    <div class="header">
        <h1 class="header-title">CFA Expert Assistant</h1>
        <div class="header-subtitle">Robust document processing for CFA materials</div>
    </div>
    """, unsafe_allow_html=True)

def progress_dashboard():
    """Processing status dashboard"""
    if st.session_state.kb_processing:
        with st.expander("📊 Processing Dashboard", expanded=True):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric("Files Processed", st.session_state.files_processed)
                st.metric("Files Failed", st.session_state.files_failed)
                
            with col2:
                st.metric("Total Chunks", st.session_state.total_chunks)
                st.caption(f"Stage: {st.session_state.progress_stage}")
                
            # Progress bar
            progress_value = st.session_state.progress / 100.0
            st.progress(progress_value, text=f"{st.session_state.progress:.1f}%")
                
        # Show file status
        if st.session_state.file_status:
            st.subheader("📁 File Processing Status")
            for filename, status in st.session_state.file_status.items():
                col1, col2 = st.columns([3, 9])
                with col1:
                    if status == "processing":
                        st.info("🔄 Processing")
                    elif status == "completed":
                        st.success("✅ Complete")
                    elif status == "failed":
                        st.error("❌ Failed")
                with col2:
                    st.write(filename)
            
        # Show error details
        if st.session_state.file_errors:
            with st.expander("⚠️ Error Details", expanded=False):
                for filename, error in st.session_state.file_errors.items():
                    st.error(f"**{filename}**: {error}")

def file_upload_section():
    """File upload handling"""
    uploaded_files = st.file_uploader(
        "📄 Upload CFA Materials (PDF files)",
        type="pdf",
        accept_multiple_files=True,
        help="Select multiple PDF files to build your knowledge base",
        key="file_uploader"
    )
    
    if uploaded_files:
        # Validate files
        total_size_mb = sum(u.size for u in uploaded_files) / (1024*1024)
        
        if total_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"⚠️ Total size ({total_size_mb:.1f}MB) exceeds limit ({MAX_FILE_SIZE_MB}MB)")
            return []
        
        # Show file summary
        st.success(f"✅ {len(uploaded_files)} files selected ({total_size_mb:.1f}MB total)")
        
        # Show file list
        with st.expander("📋 Selected Files", expanded=False):
            for i, file in enumerate(uploaded_files):
                file_size_mb = file.size / (1024*1024)
                st.write(f"{i+1}. {file.name} ({file_size_mb:.1f}MB)")
    
    return uploaded_files

def knowledge_operations(uploaded_files):
    """Build/Reset operations"""
    col1, col2, col3 = st.columns([4, 3, 3])
    
    with col1:
        build_disabled = (not uploaded_files or 
                         st.session_state.kb_processing)
        build_button = st.button(
            "🏗️ Build Knowledge Base",
            disabled=build_disabled,
            help="Process selected files and build the knowledge base",
            type="primary" if not build_disabled else "secondary"
        )

    with col2:
        model_options = {
            "gemini-1.5-flash": "Gemini 1.5 Flash",
            "gemini-1.0-pro": "Gemini 1.0 Pro",
            "gemini-pro": "Gemini Pro"
        }
        st.selectbox(
            "🤖 Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            key="model",
            help="Select the Generative AI model to use"
        )

    with col3:
        if st.button("🔄 Reset System", type="secondary",
                    help="Clear the knowledge base and start fresh",
                    disabled=st.session_state.kb_processing):
            try:
                # Clear vector DB
                if os.path.exists("./vector_db"):
                    shutil.rmtree("./vector_db")
                # Reset all state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                init_session_state()
                st.success("✅ System reset complete")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Reset failed: {str(e)}")

    # Handle build operation
    if build_button:
        with st.spinner("Building knowledge base..."):
            success = build_knowledge_base(uploaded_files)
            if success:
                st.success("✅ Knowledge base built successfully!")
            else:
                st.error("❌ Knowledge base build failed. Check the error details.")

# Chat interface with proper error handling
def chat_interface():
    """Interactive chat interface"""
    chat_container = st.container()
    
    with chat_container:
        # Display conversation history
        for msg in get_state('conversation', []):
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(msg["content"])
                    if "sources" in msg:
                        st.caption(f"Sources: {', '.join(sorted(msg['sources']))}")
        
        # Handle new messages
        if prompt := st.chat_input("Ask a CFA-related question...", 
                                 disabled=not get_state('db_exists', False)):
            if not prompt.strip():
                st.warning("Please enter a valid question")
                return
                
            # Add user message
            st.chat_message("user").write(prompt)
            st.session_state['conversation'].append({
                "role": "user",
                "content": prompt
            })
            
            # Generate and display response
            answer = "Error occurred"
            sources = set()
            
            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base..."):
                    try:
                        set_state('processing', True)
                        result = handle_user_query(prompt)
                        if isinstance(result, tuple) and len(result) >= 2:
                            answer = result[0]
                            sources = result[1]
                        st.write(answer)
                        if sources:
                            st.caption(f"Sources: {', '.join(sorted(sources))}")
                    except Exception as e:
                        error_msg = f"Error processing your question: {str(e)}"
                        st.error(error_msg)
                        logger.error(error_msg)
                        answer = error_msg
                        sources = set()
                    finally:
                        set_state('processing', False)
                    
            # Save assistant response
            st.session_state['conversation'].append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

def main():
    """Main application entry point"""
    # Initialize session state
    if "first_run" not in st.session_state:
        st.session_state["first_run"] = True
        init_session_state()
        
    if get_state("first_run", True):
        header_section()
        set_state("first_run", False)
    else:
        header_section()
    
    # Show any system-wide errors
    if error_msg := get_state('error_message'):
        st.error(error_msg)
    
    # Main interface sections
    with st.sidebar:
        st.subheader("📚 Document Management")
        uploaded_files = file_upload_section()
        st.divider()
        knowledge_operations(uploaded_files)
        
        # Show DB status
        if get_state('db_exists', False):
            st.success("✅ Knowledge base is ready")
        else:
            st.info("ℹ️ Upload PDFs and build the knowledge base to start")
    
    # Progress tracking
    if get_state('kb_processing', False):
        progress_dashboard()
    
    # Chat interface
    if not get_state('kb_processing', False):
        chat_interface()

if __name__ == "__main__":
    main()