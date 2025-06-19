import os
import glob
import sys
import tempfile
from typing import Optional, List, Any
from pathlib import Path
import streamlit as st
from langchain_community.document_loaders import (
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
try:
    from langchain.chains import RetrievalQA
except ImportError:
    from langchain.chains.retrieval_qa import RetrievalQA
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from PIL import Image
import pytesseract
import sqlite3
import google.generativeai as genai
from pydantic import Field
import hashlib
import json

# Configure Poppler path
POPPLER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'poppler', 'bin')
if os.path.exists(POPPLER_PATH):
    os.environ["PATH"] = POPPLER_PATH + os.pathsep + os.environ["PATH"]

# Import PDF conversion library
try:
    from pdf2image import convert_from_path, pdfinfo_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    st.warning("pdf2image not available. PDF processing will be disabled.")

# Configure Tesseract path
def setup_tesseract():
    """Setup Tesseract OCR with fallback paths."""
    try:
        pytesseract.get_tesseract_version()
        return True
    except pytesseract.TesseractNotFoundError:
        # Common Tesseract paths
        tesseract_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract"
        ]
        
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                try:
                    pytesseract.get_tesseract_version()
                    return True
                except:
                    continue
        return False

TESSERACT_AVAILABLE = setup_tesseract()

def check_dependencies():
    """Check if required dependencies are installed and configured."""
    missing_deps = []
    warnings = []
    
    # Check Tesseract
    if not TESSERACT_AVAILABLE:
        missing_deps.append("Tesseract OCR")
    
    # Check pdf2image
    if not PDF2IMAGE_AVAILABLE:
        warnings.append("pdf2image library not available - PDF processing disabled")
    
    # Check Poppler for Windows (only if pdf2image is available)
    if PDF2IMAGE_AVAILABLE and sys.platform == "win32":
        try:
            # Simple test without requiring actual PDF files
            if not os.path.exists(POPPLER_PATH):
                warnings.append("Poppler not found in expected location - PDF processing may fail")
        except Exception as e:
            warnings.append(f"Poppler check failed: {e}")
    
    return missing_deps, warnings

def get_file_hash(file_content):
    """Generate hash for file content to use as cache key."""
    return hashlib.md5(file_content).hexdigest()

# Custom loader for PDF files using OCR
def load_pdf_with_ocr(file_path: str, max_pages: int = 50) -> List[Document]:
    """Load PDF with OCR, with limits to prevent hanging."""
    if not PDF2IMAGE_AVAILABLE or not TESSERACT_AVAILABLE:
        st.error("PDF processing requires both pdf2image and Tesseract OCR to be installed.")
        return []
    
    documents = []
    try:
        st.info(f"Converting PDF '{os.path.basename(file_path)}' to images...")
        
        # Get PDF info first to check page count
        try:
            pdf_info = pdfinfo_from_path(file_path)
            total_pages = pdf_info.get('Pages', 0)
            
            if total_pages > max_pages:
                st.warning(f"PDF has {total_pages} pages. Processing only the first {max_pages} pages to prevent timeout.")
                pages_to_process = list(range(1, min(max_pages + 1, total_pages + 1)))
            else:
                pages_to_process = None
                
        except Exception as e:
            st.warning(f"Could not get PDF info: {e}. Processing with default settings...")
            pages_to_process = None
        
        # Convert PDF to images with page limit
        images = convert_from_path(
            file_path, 
            first_page=1 if pages_to_process else None,
            last_page=max_pages if pages_to_process else None,
            dpi=150,  # Lower DPI for faster processing
            fmt='jpeg'  # JPEG for smaller memory footprint
        )
        
        if not images:
            st.error("No images could be extracted from the PDF.")
            return []
        
        page_progress = st.progress(0)
        status_text = st.empty()

        for i, image in enumerate(images):
            if i >= max_pages:  # Additional safety check
                break
                
            page_progress.progress((i + 1) / len(images))
            status_text.text(f"Processing page {i+1}/{len(images)} of {os.path.basename(file_path)} with OCR...")
            
            try:
                # Optimize image for OCR
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Extract text with timeout protection
                extracted_text = pytesseract.image_to_string(
                    image, 
                    config='--psm 1 --oem 3',  # OCR configuration
                    timeout=30  # 30 second timeout per page
                )
                
                text = str(extracted_text).strip() if extracted_text else ""
                
                if text and len(text) > 10:  # Only add if substantial text found
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": str(file_path), "page": i + 1}
                        )
                    )
                    
            except Exception as e:
                st.warning(f"Could not process page {i+1}: {str(e)[:100]}...")
                continue
        
        status_text.empty()
        page_progress.empty()
                
    except Exception as e:
        st.error(f"Error processing PDF file {os.path.basename(file_path)}: {str(e)[:200]}...")
        return []
        
    return documents

# Custom loader for .db files (SQLite assumed)
def load_from_db(file_path):
    """Load data from SQLite database with error handling."""
    documents = []
    try:
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()
        
        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            st.warning(f"No tables found in database {os.path.basename(file_path)}")
            conn.close()
            return []
        
        for table in tables:
            table_name = table[0]
            try:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 1000")  # Limit rows to prevent memory issues
                rows = cursor.fetchall()
                
                for i, row in enumerate(rows):
                    text = ' '.join([str(item) for item in row if item is not None])
                    if text.strip():
                        documents.append(
                            Document(
                                page_content=text, 
                                metadata={"source": file_path, "table": table_name, "row": i+1}
                            )
                        )
            except Exception as e:
                st.warning(f"Error reading table {table_name}: {e}")
                continue
                
        conn.close()
        
    except Exception as e:
        st.error(f"Error processing database file {os.path.basename(file_path)}: {e}")
        return []
        
    return documents

# Custom loader for image files using OCR
def load_from_image(file_path: str) -> List[Document]:
    """Load text from image using OCR."""
    if not TESSERACT_AVAILABLE:
        st.error("Image processing requires Tesseract OCR to be installed.")
        return []
    
    try:
        st.info(f"Processing image '{os.path.basename(file_path)}' with OCR...")
        
        image = Image.open(file_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Extract text with timeout
        text = pytesseract.image_to_string(
            image, 
            config='--psm 1 --oem 3',
            timeout=30
        )
        
        if isinstance(text, (bytes, bytearray)):
            text = text.decode('utf-8')
            
        text = str(text).strip()
        
        if not text or len(text) < 5:
            st.warning(f"No meaningful text found in {os.path.basename(file_path)}")
            return []
            
        return [Document(page_content=text, metadata={"source": str(file_path)})]
        
    except Exception as e:
        st.error(f"Error processing image file {os.path.basename(file_path)}: {e}")
        return []

# Function to load documents based on file extension
def load_documents_from_file(file_path):
    """Load documents from various file types with error handling."""
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == '.pdf':
            return load_pdf_with_ocr(file_path)
        elif ext == '.csv':
            loader = CSVLoader(file_path)
            return loader.load()
        elif ext == '.xlsx':
            loader = UnstructuredExcelLoader(file_path)
            return loader.load()
        elif ext == '.pptx':
            loader = UnstructuredPowerPointLoader(file_path)
            return loader.load()
        elif ext == '.db':
            return load_from_db(file_path)
        elif ext in ['.jpg', '.png', '.jpeg']:
            return load_from_image(file_path)
        else:
            st.warning(f"Unsupported file type: {ext}")
            return []
            
    except Exception as e:
        st.error(f"Error loading {os.path.basename(file_path)}: {str(e)[:200]}...")
        return []

# Custom LLM class for Google Gemini
class GeminiLLM(LLM):
    api_key: str = Field(..., description="Google API key for Gemini")

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Add timeout and error handling
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1000,
                    temperature=0.7,
                )
            )
            
            if response and response.text:
                return str(response.text)
            else:
                return "I couldn't generate a proper response. Please try rephrasing your question."
                
        except Exception as e:
            error_msg = str(e)
            st.error(f"Gemini API Error: {error_msg[:100]}...")
            return f"Sorry, I encountered an error: {error_msg[:100]}... Please check your API key and try again."

    @property
    def _llm_type(self) -> str:
        return "gemini"

# Function to initialize the QA chain with better caching
def create_qa_chain(documents, api_key):
    """Create QA chain with improved error handling and resource management."""
    try:
        st.info("Filtering and preparing documents...")
        
        # Filter out empty documents
        valid_documents = []
        for doc in documents:
            if doc.page_content and doc.page_content.strip() and len(doc.page_content.strip()) > 10:
                valid_documents.append(doc)
        
        if not valid_documents:
            st.error("No valid documents found after filtering. Please check your files.")
            return None
        
        st.info(f"Processing {len(valid_documents)} valid document sections...")
        
        # Split documents with reasonable chunk size
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller chunks
            chunk_overlap=100,
            length_function=len,
        )
        
        splits = text_splitter.split_documents(valid_documents)
        
        if not splits:
            st.error("Could not create text chunks from documents.")
            return None
        
        st.info(f"Created {len(splits)} text chunks.")
        
        # Create embeddings with progress indicator
        with st.spinner("Creating embeddings and vector store..."):
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name='all-MiniLM-L6-v2',
                    model_kwargs={'device': 'cpu'},  # Force CPU to avoid GPU issues
                    encode_kwargs={'normalize_embeddings': True}
                )
                
                # Create vector store in batches to prevent memory issues
                batch_size = 50
                if len(splits) > batch_size:
                    st.info(f"Processing embeddings in batches of {batch_size}...")
                    vectorstore = FAISS.from_documents(splits[:batch_size], embeddings)
                    
                    for i in range(batch_size, len(splits), batch_size):
                        batch_end = min(i + batch_size, len(splits))
                        batch_splits = splits[i:batch_end]
                        batch_vectorstore = FAISS.from_documents(batch_splits, embeddings)
                        vectorstore.merge_from(batch_vectorstore)
                        st.info(f"Processed {batch_end}/{len(splits)} chunks...")
                else:
                    vectorstore = FAISS.from_documents(splits, embeddings)
                
                retriever = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}  # Limit retrieved documents
                )
                
            except Exception as e:
                st.error(f"Error creating embeddings: {e}")
                return None

        # Initialize the Gemini LLM
        try:
            llm = GeminiLLM(api_key=api_key)
        except Exception as e:
            st.error(f"Error initializing Gemini LLM: {e}")
            return None

        # Set up the RAG model
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff", 
                retriever=retriever, 
                return_source_documents=True,
                verbose=False
            )
            
            st.success("✅ Model is ready to answer your questions!")
            return qa_chain
            
        except Exception as e:
            st.error(f"Error creating QA chain: {e}")
            return None
            
    except Exception as e:
        st.error(f"Unexpected error in create_qa_chain: {e}")
        return None

# --- STREAMLIT UI ---
def main():
    st.set_page_config(
        page_title="Document Q&A with Gemini", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📄 Document Q&A with Google Gemini")
    
    # Initialize session state
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

    # --- DEPENDENCY CHECK ---
    missing_deps, warnings = check_dependencies()
    
    if missing_deps:
        st.error(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        
        if "Tesseract OCR" in missing_deps:
            st.markdown("""
            **To install Tesseract OCR:**
            - **Windows**: Download from [here](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH
            - **Mac**: `brew install tesseract`
            - **Linux**: `sudo apt-get install tesseract-ocr`
            """)
            
        return  # Don't proceed if critical dependencies are missing
    
    if warnings:
        for warning in warnings:
            st.warning(f"⚠️ {warning}")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "🔑 Google API Key", 
            type="password", 
            help="Get your key from Google AI Studio (https://aistudio.google.com/)"
        )
        
        st.markdown("---")
        
        uploaded_files = st.file_uploader(
            "📁 Upload Documents", 
            type=['pdf', 'csv', 'xlsx', 'pptx', 'db', 'jpg', 'png', 'jpeg'],
            accept_multiple_files=True,
            help="Supported: PDF, CSV, Excel, PowerPoint, SQLite DB, Images"
        )
        
        # Processing options
        st.subheader("Processing Options")
        max_pdf_pages = st.slider("Max PDF pages to process", 5, 100, 50, 5)
        
        process_button = st.button("🚀 Process Documents", type="primary")
        
        if st.button("🗑️ Clear Session"):
            st.session_state.qa_chain = None
            st.session_state.processed_files = []
            st.rerun()

        # Show processed files
        if st.session_state.processed_files:
            st.subheader("📋 Processed Files")
            for filename in st.session_state.processed_files:
                st.text(f"✅ {filename}")

    # --- MAIN CONTENT ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 How to Use
        1. **Enter your Google API key** in the sidebar
        2. **Upload documents** (PDF, CSV, Excel, PowerPoint, SQLite DB, Images)
        3. **Click "Process Documents"** and wait for completion
        4. **Ask questions** about your documents below
        
        ### 📝 Supported Features
        - **PDF**: OCR text extraction from scanned documents
        - **Images**: Text extraction using OCR
        - **Structured Data**: CSV, Excel, PowerPoint files
        - **Databases**: SQLite database files
        """)
    
    with col2:
        st.info("""
        **💡 Tips for better results:**
        - Use clear, specific questions
        - Mention document names if relevant
        - Ask follow-up questions for clarification
        """)

    # Process documents
    if process_button and uploaded_files:
        if not api_key:
            st.error("❌ Please enter your Google API Key first.")
        else:
            documents = []
            processed_files = []
            
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                for i, uploaded_file in enumerate(uploaded_files):
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Save uploaded file
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    try:
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        
                        # Load documents
                        docs = load_documents_from_file(temp_path)
                        
                        if docs:
                            documents.extend(docs)
                            processed_files.append(uploaded_file.name)
                            st.success(f"✅ {uploaded_file.name}: {len(docs)} sections loaded")
                        else:
                            st.warning(f"⚠️ {uploaded_file.name}: No content extracted")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)[:100]}...")
                        continue
            
            progress_bar.empty()
            status_text.empty()
            
            if documents:
                st.info(f"📊 Total sections loaded: {len(documents)}")
                qa_chain = create_qa_chain(documents, api_key)
                
                if qa_chain:
                    st.session_state.qa_chain = qa_chain
                    st.session_state.processed_files = processed_files
                else:
                    st.error("❌ Failed to create QA chain. Please try again.")
            else:
                st.error("❌ No documents could be processed. Please check your files.")

    # Q&A Interface
    st.markdown("---")
    
    if st.session_state.qa_chain:
        st.subheader("💬 Ask Questions")
        
        # Question input
        query = st.text_input(
            "What would you like to know about your documents?",
            placeholder="e.g., What are the main findings in the report?",
            key="question_input"
        )
        
        # Sample questions
        with st.expander("💡 Sample Questions"):
            sample_questions = [
                "What is the main topic discussed in the documents?",
                "Can you summarize the key findings?",
                "What are the important dates mentioned?",
                "List the main recommendations provided.",
                "What are the key statistics or numbers mentioned?"
            ]
            for q in sample_questions:
                if st.button(q, key=f"sample_{hash(q)}"):
                    st.session_state.question_input = q
                    st.rerun()
        
        if query:
            with st.spinner("🤔 Thinking..."):
                try:
                    result = st.session_state.qa_chain({"query": query})
                    response = result.get('result', 'No response generated.')
                    source_docs = result.get('source_documents', [])
                    
                    # Display answer
                    st.subheader("📝 Answer")
                    st.write(response)
                    
                    # Display sources
                    if source_docs:
                        with st.expander(f"📚 Sources ({len(source_docs)} found)"):
                            for i, doc in enumerate(source_docs, 1):
                                source = doc.metadata.get('source', 'Unknown')
                                page = doc.metadata.get('page', 'N/A')
                                
                                st.markdown(f"**Source {i}:** `{os.path.basename(source)}` | **Page:** `{page}`")
                                
                                # Show content preview
                                content_preview = doc.page_content[:300]
                                if len(doc.page_content) > 300:
                                    content_preview += "..."
                                st.markdown(f"> {content_preview}")
                                
                                if i < len(source_docs):
                                    st.markdown("---")
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)[:200]}...")
                    
    else:
        st.info("👆 Please upload and process your documents using the sidebar to begin asking questions.")

if __name__ == "__main__":
    main()