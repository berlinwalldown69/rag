import os
import gc
import time
import re
import traceback
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize session state
def init_session_state():
    keys_to_init = {
        'conversation': [],
        'processing': False,
        'db_exists': False,
        'uploaded_files': [],
        'model_name': "gemini-1.5-flash",
        'sources': set(),
        'first_load': True
    }
    for key, default in keys_to_init.items():
        if key not in st.session_state:
            st.session_state[key] = default

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("🔑 GEMINI_API_KEY not found in .env file or environment variables")
    st.stop()

# --- Configuration ---
CONFIG = {
    "DB_PATH": "./cfa_chroma_db",
    "RAG_TOPK": 5,
    "CHUNK_SIZE": 1000,
    "CHUNK_OVERLAP": 100,
    "MAX_ANSWER_LENGTH": 3000,
    "LLM_TEMPERATURE": 0.35,
    "LLM_MAX_TOKENS": 2000  # Reduced to improve reliability
}

# --- Helper Functions ---
def create_file_path(filename):
    """Ensure DB path exists and create a safe file path"""
    os.makedirs(CONFIG["DB_PATH"], exist_ok=True)
    return os.path.join(CONFIG["DB_PATH"], re.sub(r'[^a-zA-Z0-9_.]', '_', filename))

def wrap_text(text: str, width: int = 90) -> str:
    """Wrap long text for better readability"""
    if not text:
        return ""
    return "\n".join([text[i:i+width] for i in range(0, len(text), width)])

# --- Document Processing ---
def safe_extract_text(pdf_path: str) -> str:
    """Robust text extraction with fallback mechanisms"""
    try:
        # Try with PyPDF first
        loader = PyPDFLoader(pdf_path)
        text = " ".join([page.page_content for page in loader.load()])
        if text.strip():
            return text
    except Exception as e:
        st.warning(f"PyPDF extraction failed: {str(e)}")
    
    try:
        # Fallback to unstructured loader
        loader = UnstructuredPDFLoader(pdf_path)
        data = loader.load()
        if data and data[0].page_content:
            return data[0].page_content
    except Exception as e:
        st.error(f"Unstructured extraction failed: {str(e)}")
    
    return ""

def process_pdf_chunk(pdf_path: str, vector_db, filename: str) -> int:
    """Process a single PDF with memory constraints"""
    try:
        text = safe_extract_text(pdf_path)
        if not text.strip():
            st.error(f"⛔ Failed to extract text from: {filename}")
            return 0
            
        # Preprocessing
        text = re.sub('\n+', ' ', text)  # Remove excessive newlines
        text = re.sub('\s+', ' ', text)   # Excess spaces
        text = text[:500000]  # Limit to 500K characters
        
        # Smart chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["CHUNK_SIZE"],
            chunk_overlap=CONFIG["CHUNK_OVERLAP"],
            length_function=len
        )
        
        chunks = splitter.split_text(text)
        metadatas = [{"source": filename} for _ in chunks]
        
        # Add to vector store
        vector_db.add_texts(texts=chunks, metadatas=metadatas)
        return len(chunks)
        
    except Exception as e:
        st.error(f"Error processing {filename}: {traceback.format_exc()}")
        return 0

# --- Knowledge Base Functions ---
def build_knowledge_base(uploaded_files=[]):
    """Build or update knowledge base"""
    if not uploaded_files:
        return 0
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=API_KEY,
            task_type="retrieval_document"
        )
        
        # Initialize Chroma with error handling
        try:
            vector_db = Chroma(
                persist_directory=CONFIG["DB_PATH"],
                embedding_function=embeddings,
                collection_name="cfa_knowledge"
            )
        except:
            # If collection doesn't exist
            vector_db = Chroma.from_texts(
                texts=[""],
                embedding=embeddings,
                persist_directory=CONFIG["DB_PATH"],
                collection_name="cfa_knowledge"
            )
        
        total_chunks = 0
        source_files = set()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each file
        for i, uploaded_file in enumerate(uploaded_files):
            file_path = create_file_path(uploaded_file.name)
            
            # Save uploaded file to disk
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            start_time = time.time()
            chunks_added = process_pdf_chunk(file_path, vector_db, uploaded_file.name)
            total_chunks += chunks_added
            
            if chunks_added > 0:
                source_files.add(uploaded_file.name)
                st.session_state.uploaded_files.append(uploaded_file.name)
            
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"📖 Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name} ({chunks_added} chunks)")
            
            # Memory management
            gc.collect()
            time.sleep(0.2)  # Reduce CPU load
            
        # Finalize KB
        vector_db.persist()
        st.session_state.db_exists = True
        st.session_state.sources = source_files
        
        progress_bar.empty()
        if total_chunks > 0:
            status_text.success(f"✅ Knowledge base built with {total_chunks} chunks from {len(source_files)} files!")
        else:
            status_text.warning("⚠️ No content added to knowledge base")
    
    except Exception as e:
        st.error(f"❌ Critical error: {traceback.format_exc()}")

# --- Reasoning Engine ---
def modern_reasoning_engine(question: str, context: str):
    """Advanced reasoning pipeline with error handling"""
    SYSTEM_PROMPT = """
As a CFA expert, analyze this question using ONLY the provided context. Perform:
1. Concept identification (CFA level/topic)
2. Key term definitions
3. Step-by-step explanation
4. Relevant formulas/theories
5. Practical applications

CONTEXT:
{context}

QUESTION:
{question}"""

    try:
        # Initialize model
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=API_KEY,
            temperature=CONFIG["LLM_TEMPERATURE"],
            max_output_tokens=CONFIG["LLM_MAX_TOKENS"]
        )
        
        # Format prompt with truncation to prevent overflow
        safe_context = context[:20000]  # Limit context size
        prompt = SYSTEM_PROMPT.format(context=safe_context, question=question[:500])
        
        # Get the response
        response = llm.invoke(prompt).content
        return response[:CONFIG["MAX_ANSWER_LENGTH"]]
    except Exception as e:
        traceback.print_exc()
        return "⚠️ System error: Answer could not be generated"

# --- Query Handling ---
def retrieve_context(question: str):
    """Retrieve relevant context from vector DB"""
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
        
        # Retrieve documents
        docs = vector_db.similarity_search(question, k=CONFIG["RAG_TOPK"])
        
        # Build context
        context_parts = []
        sources = set()
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown Source")
            sources.add(source)
            context_parts.append(f"SOURCE {i+1}: {source}\n{doc.page_content[:800]}\n")
        
        return "\n".join(context_parts), sources
        
    except Exception as e:
        traceback.print_exc()
        return "", set()

def process_query(question: str):
    """End-to-end query processing with timing"""
    start_time = time.time()
    
    # Retrieve context
    context, sources = retrieve_context(question)
    
    if not context:
        return "⚠️ No relevant information found in knowledge base", set(), 0
    
    try:
        # Get reasoned response
        answer = modern_reasoning_engine(question, context)
        return answer, sources, time.time() - start_time
    except Exception as e:
        traceback.print_exc()
        return "⚠️ Answer generation failed", set(), time.time() - start_time
    
# --- Streamlit UI Components ---
def render_message(role, content, sources=set()):
    """Render chat message with source citations"""
    avatar = "💼" if role == "assistant" else "🧑"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        
        if sources:
            with st.expander("📚 Source Documents"):
                for source in sorted(sources):
                    st.write(f"- {source}")

# --- Streamlit App ---
# Page setup
st.set_page_config(
    page_title="CFA Expert Assistant",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "CFA Assistant v2.0 - RAG System for CFA Curriculum"
    }
)

# Custom CSS for cleaner UI
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    .st-chat { border-radius: 15px; }
    .stChatMessage { padding: 12px 16px; }
    [data-testid="stSidebar"] { background-color: #1e3a66; color: white; }
    .sidebar-title { color: white !important; }
    .instructions { background-color: #e9f7ef; padding: 15px; border-radius: 10px; }
    .stProgress > div > div > div { background-color: #1e88e5 !important; }
    .stMarkdown { color: #333; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
init_session_state()

# Sidebar Configuration (always visible)
with st.sidebar:
    st.markdown("<h1 class='sidebar-title'>CFA Knowledge Base</h1>", unsafe_allow_html=True)
    
    with st.expander("🎯 About This Tool"):
        st.info("""
        **CFA Expert Assistant** helps you study with:
        - 📚 PDF knowledge base
        - 💡 Deep reasoning engine
        - ✅ Source citations
        - ⚡ Efficient RAG architecture
        """)
    
    st.markdown("---")
    st.subheader("1️⃣ Upload Syllabus")
    
    uploaded_files = st.file_uploader(
        "Upload CFA PDF Files",
        type="pdf",
        accept_multiple_files=True,
        help="Curriculum materials, textbooks, practice questions"
    )
    
    if st.button("🏗️ Build Knowledge Base", use_container_width=True):
        if uploaded_files:
            with st.spinner("Indexing documents..."):
                build_knowledge_base(uploaded_files)
        else:
            st.warning("Please upload PDF files first")
    
    if st.session_state.uploaded_files:
        st.markdown("---")
        st.subheader("📚 Current Materials")
        for i, file in enumerate(st.session_state.uploaded_files[:10]):
            st.caption(f"- {file}")
        if len(st.session_state.uploaded_files) > 10:
            st.caption(f"... and {len(st.session_state.uploaded_files)-10} more")
    
    st.markdown("---")
    st.caption("For best results: Use official CFA curriculum PDFs. Processing may take several minutes per file.")

# Main content area
st.markdown("<h1 style='text-align: center;'>CFA Expert Assistant</h1>", unsafe_allow_html=True)
st.caption("<p style='text-align: center;'>Powered by Gemini AI • Built with Streamlit</p>", unsafe_allow_html=True)

# Welcome message on first load
if st.session_state.first_load:
    with st.chat_message("assistant", avatar="💼"):
        st.markdown("👋 Welcome to your CFA study assistant! \n\n1. Upload curriculum PDFs in the sidebar\n2. Build your knowledge base\n3. Ask questions about CFA topics")
    st.session_state.first_load = False

# Display conversation history
for message in st.session_state.conversation:
    render_message(**message)

# User input with improved visual priority
query = st.chat_input("Ask any CFA-level question...", key="chat_input")

if query and not st.session_state.processing:
    # Add user message to history
    st.session_state.conversation.append({"role": "user", "content": query})
    render_message("user", query)

    st.session_state.processing = True
    with st.spinner("🔍 Analyzing CFA curriculum..."):
        try:
            # Process query
            with st.chat_message("assistant", avatar="💼"):
                placeholder = st.empty()
                placeholder.markdown("▌")
                
                answer, sources, pt = process_query(query)
                
                # Display response
                placeholder.markdown(answer)
                st.session_state.conversation.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
                # Show sources
                if sources:
                    with st.expander("📚 Source Documents"):
                        for source in sorted(sources):
                            st.write(f"- {source}")
                
                # Add processing time
                st.caption(f"⏱️ Generated in {pt:.1f} seconds")
        except Exception as e:
            st.error(f"System error: {str(e)}")
            st.session_state.conversation.append({
                "role": "assistant",
                "content": "⚠️ Temporary system issue - please try again"
            })
            
    st.session_state.processing = False
    gc.collect()
