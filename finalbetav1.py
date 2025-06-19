import os
import gc
import time
import re
import traceback
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from collections import deque

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize session state for all necessary keys
def init_session_state():
    """Initializes session state variables if they don't exist."""
    keys_to_init = {
        'conversation': [],
        'processing': False,
        'db_exists': False,
        'uploaded_files': [],
        'model_name': "gemini-1.5-flash",
        'sources': set(),
        'first_load': True,
        'chat_history': deque(maxlen=10) # Memory: Stores last 5 Q&A pairs
    }
    for key, default in keys_to_init.items():
        if key not in st.session_state:
            st.session_state[key] = default

# Load environment variables
try:
    load_dotenv()
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        st.error("🔑 GEMINI_API_KEY not found. Please set it in your .env file or environment variables.")
        st.stop()
except Exception as e:
    st.error(f"Error loading environment variables: {e}")
    st.stop()

# --- Configuration ---
CONFIG = {
    "DB_PATH": "./cfa_chroma_db_v3",
    "RAG_TOPK": 5,
    "CHUNK_SIZE": 1500, # Increased for more context per chunk
    "CHUNK_OVERLAP": 250, # Increased overlap
    "MAX_ANSWER_LENGTH": 4000,
    "LLM_TEMPERATURE": 0.4, # Slightly higher for more nuanced answers
    "LLM_MAX_TOKENS": 4096
}

# --- Helper Functions ---
def create_file_path(filename: str) -> str:
    """Creates a sanitized, absolute path for a file within the DB directory."""
    db_path = Path(CONFIG["DB_PATH"]).resolve()
    db_path.mkdir(parents=True, exist_ok=True)
    safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    return str(db_path / safe_filename)

# --- Document Processing ---
def safe_extract_text(pdf_path: str) -> str:
    """
    Robustly extracts text from a PDF using multiple methods as fallback.
    Returns extracted text or an empty string if all methods fail.
    """
    text = ""
    try:
        # 1. PyPDFLoader: Fast and structured
        loader = PyPDFLoader(pdf_path, extract_images=False)
        pages = loader.load()
        text = " ".join([page.page_content for page in pages])
        if text.strip():
            return text
        st.warning(f"PyPDFLoader extracted no text from {Path(pdf_path).name}. Trying fallback.")
    except Exception as e:
        st.warning(f"PyPDFLoader failed on {Path(pdf_path).name}: {e}. Trying fallback.")

    try:
        # 2. UnstructuredPDFLoader: More robust for complex layouts
        loader = UnstructuredPDFLoader(pdf_path, mode="elements")
        data = loader.load()
        if data and data[0].page_content:
            return data[0].page_content
        st.error(f"UnstructuredPDFLoader also failed to extract text from {Path(pdf_path).name}.")
    except Exception as e:
        st.error(f"UnstructuredPDFLoader failed critically on {Path(pdf_path).name}: {e}")

    return ""

def process_pdf(pdf_path: str, vector_db, filename: str) -> int:
    """
    Processes a single PDF: extracts text, splits it into chunks, and adds to the vector store.
    Includes more aggressive text cleaning.
    """
    try:
        text = safe_extract_text(pdf_path)
        if not text.strip():
            st.error(f"⛔ Failed to extract any text from: {filename}")
            return 0

        # Aggressive text cleaning
        text = text.lower() # Normalize to lowercase
        text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
        text = re.sub(r'(\n\s*){2,}', '\n', text) # Remove multiple blank lines
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text) # Remove non-printable chars

        if not text.strip():
            st.warning(f"Text for {filename} became empty after cleaning.")
            return 0

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["CHUNK_SIZE"],
            chunk_overlap=CONFIG["CHUNK_OVERLAP"],
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " ", ""] # More granular separators
        )

        chunks = text_splitter.split_text(text)
        if not chunks:
            st.warning(f"Could not split {filename} into any chunks.")
            return 0

        metadatas = [{"source": filename} for _ in chunks]
        vector_db.add_texts(texts=chunks, metadatas=metadatas)
        return len(chunks)

    except Exception:
        st.error(f"An unexpected error occurred while processing {filename}:\n{traceback.format_exc()}")
        return 0

# --- Knowledge Base Functions ---
def build_knowledge_base(uploaded_files):
    """Builds or updates the ChromaDB knowledge base from uploaded PDF files."""
    if not uploaded_files:
        st.warning("No files were uploaded.")
        return

    st.session_state.processing = True
    total_chunks = 0
    processed_files = set()
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=API_KEY,
            task_type="retrieval_document"
        )
        vector_db = Chroma(
            persist_directory=CONFIG["DB_PATH"],
            embedding_function=embeddings,
            collection_name="cfa_knowledge_v3"
        )

        progress_bar = st.progress(0, "Initializing knowledge base build...")
        status_text = st.empty()

        for i, uploaded_file in enumerate(uploaded_files):
            file_path = create_file_path(uploaded_file.name)
            if uploaded_file.name in st.session_state.uploaded_files:
                status_text.text(f"Skipping already processed file: {uploaded_file.name}")
                continue

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            status_text.text(f"📖 Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}...")
            chunks_added = process_pdf(file_path, vector_db, uploaded_file.name)

            if chunks_added > 0:
                total_chunks += chunks_added
                processed_files.add(uploaded_file.name)
                st.session_state.uploaded_files.append(uploaded_file.name)

            progress_bar.progress((i + 1) / len(uploaded_files))
            gc.collect() # Force garbage collection after each file

        vector_db.persist()
        st.session_state.db_exists = True
        st.session_state.sources.update(processed_files)

        if total_chunks > 0:
            status_text.success(f"✅ Knowledge base updated with {total_chunks} new chunks from {len(processed_files)} files!")
        else:
            status_text.warning("⚠️ No new content was added to the knowledge base.")

    except Exception:
        st.error(f"❌ A critical error occurred during knowledge base creation:\n{traceback.format_exc()}")
    finally:
        st.session_state.processing = False

# --- Reasoning Engine with Memory ---
def format_chat_history(history: deque) -> str:
    """Formats the chat history for the LLM prompt."""
    if not history:
        return "No previous conversation history."
    
    formatted_history = []
    for message in history:
        role = "Student" if message["role"] == "user" else "Tutor"
        formatted_history.append(f"{role}: {message['content']}")
    
    return "Recent Conversation History (for context):\n" + "\n\n".join(formatted_history)


def modern_reasoning_engine(question: str, context: str, chat_history: deque):
    """
    More advanced reasoning pipeline that includes conversational memory.
    """
    history_str = format_chat_history(chat_history)

    # This is a much more robust prompt designed to guide the LLM's behavior
    SYSTEM_PROMPT = """
You are an expert CFA (Chartered Financial Analyst) Tutor AI. Your primary role is to help a student understand complex financial concepts from their curriculum.

**Your Persona:**
- **Socratic & Encouraging:** Don't just give the answer. Guide the student. Ask clarifying questions. For example, if they ask for a formula, provide it, but then ask "Can you explain what each component of that formula represents?"
- **Precise & Structured:** Provide answers that are clear, well-organized, and directly address the student's question. Use markdown for formatting (bolding, lists, etc.) to improve readability.
- **Context-Bound:** Your knowledge is STRICTLY LIMITED to the **'Provided Context'** and **'Recent Conversation History'** below. NEVER use external knowledge. If the answer isn't in the provided materials, state that clearly and suggest what kind of document the student might need to upload.
- **Source-Aware:** You must implicitly use the provided context. Do not explicitly cite "Source 1" or "Source 2" in your response. Weave the information into a cohesive answer.

**Task:**
Analyze the student's **'Current Question'** using the **'Recent Conversation History'** for context and the **'Provided Context'** for information.

**Output Structure:**
1.  **Concept Identification:** Briefly state the key CFA concept, topic, and level (e.g., "This question relates to Duration and Convexity, a key concept in Level I Fixed Income.").
2.  **Direct Answer:** Provide a clear, direct answer to the student's question.
3.  **Step-by-Step Explanation:** Break down the logic, calculation, or theory into easy-to-follow steps.
4.  **Guiding Question (Optional but Recommended):** End with a question to check for understanding or encourage deeper thinking.

---
**Recent Conversation History:**
{chat_history}

---
**Provided Context from Syllabus:**
{context}

---
**Current Question:**
{question}

**Your Expert Response:**
"""

    try:
        llm = ChatGoogleGenerativeAI(
            model=st.session_state.model_name,
            google_api_key=API_KEY,
            temperature=CONFIG["LLM_TEMPERATURE"],
            max_output_tokens=CONFIG["LLM_MAX_TOKENS"],
            safety_settings={'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE'} # Adjust as needed
        )
        
        prompt_template = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
        chain = prompt_template | llm | StrOutputParser()
        
        # Truncate inputs to be safe
        safe_context = context[:25000]
        safe_question = question[:1000]

        response = chain.invoke({
            "chat_history": history_str,
            "context": safe_context,
            "question": safe_question
        })
        
        return response[:CONFIG["MAX_ANSWER_LENGTH"]]

    except Exception:
        st.error(f"Error during LLM invocation: {traceback.format_exc()}")
        return "⚠️ The reasoning engine encountered an error. The model may be unavailable or the request was invalid. Please try again."


# --- Query Handling ---
def retrieve_context(question: str):
    """Retrieves relevant context chunks from the vector database."""
    if not st.session_state.db_exists:
        return "Knowledge base not built yet. Please upload files and build it first.", set()

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
        vector_db = Chroma(
            persist_directory=CONFIG["DB_PATH"],
            embedding_function=embeddings,
            collection_name="cfa_knowledge_v3"
        )
        
        retriever = vector_db.as_retriever(search_kwargs={"k": CONFIG["RAG_TOPK"]})
        docs = retriever.invoke(question)
        
        if not docs:
            return "No relevant information was found in the uploaded documents for your question.", set()

        context_parts = []
        sources = set()
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown Source")
            sources.add(source)
            # Add more content from each chunk
            context_parts.append(f"[Content from {source}]:\n{doc.page_content}\n")
        
        return "\n---\n".join(context_parts), sources

    except Exception:
        st.error(f"Error retrieving context from database: {traceback.format_exc()}")
        return "Error accessing the knowledge base. It might be corrupted or inaccessible.", set()

def process_query(question: str, chat_history: deque):
    """End-to-end query processing including context retrieval and reasoning."""
    start_time = time.time()
    
    context, sources = retrieve_context(question)
    if not sources: # If context retrieval itself returned a message
        return context, set(), 0

    answer = modern_reasoning_engine(question, context, chat_history)
    processing_time = time.time() - start_time
    
    return answer, sources, processing_time
    
# --- Streamlit UI Components ---
def render_message(role, content, sources=None):
    """Renders a single chat message with an avatar and optional sources."""
    avatar = "💼" if role == "assistant" else "🧑‍🎓"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        if sources:
            with st.expander("📚 Sources Consulted"):
                for source in sorted(list(sources)):
                    st.caption(f"- {source}")

# --- Streamlit App ---
st.set_page_config(
    page_title="CFA Tutor AI",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more professional look
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    .st-chat { border-radius: 12px; }
    .stChatMessage { padding: 14px 18px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    [data-testid="stSidebar"] { background-color: #002366; color: white; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: white; }
    .stButton>button { border-radius: 8px; border: 2px solid #00BFFF; background-color: #00BFFF; color: white; font-weight: bold; }
    .stButton>button:hover { border-color: #009ACD; background-color: #009ACD; }
    .stProgress > div > div > div > div { background-color: #00BFFF !important; }
</style>
""", unsafe_allow_html=True)

init_session_state()

# Sidebar for controls
with st.sidebar:
    st.markdown("<h1>CFA Tutor AI</h1>", unsafe_allow_html=True)
    st.info("Your personal AI assistant for the CFA curriculum.")
    
    st.markdown("---")
    st.header("1. Upload Syllabus")
    uploaded_files = st.file_uploader(
        "Upload CFA Curriculum PDFs",
        type="pdf",
        accept_multiple_files=True,
        help="Upload curriculum sections, readings, or notes."
    )
    
    if st.button("🏗️ Build Knowledge Base", use_container_width=True, disabled=st.session_state.processing):
        if uploaded_files:
            with st.spinner("Processing documents... This may take several minutes."):
                build_knowledge_base(uploaded_files)
        else:
            st.warning("Please upload at least one PDF file.")
    
    if st.session_state.uploaded_files:
        st.markdown("---")
        st.header("📚 Loaded Materials")
        with st.expander("Click to see all loaded files"):
            for f in st.session_state.uploaded_files:
                st.caption(f"- {f}")

# Main chat interface
st.title("Chat with your CFA Tutor")
st.caption("Powered by Gemini 1.5 Flash • Enhanced with Conversational Memory")

# Initial welcome message
if st.session_state.first_load:
    st.session_state.conversation.append({
        "role": "assistant",
        "content": "👋 Welcome! I'm your AI-powered CFA Tutor.\n\n**To get started:**\n1.  Upload your CFA curriculum PDFs in the sidebar.\n2.  Click **'Build Knowledge Base'**.\n3.  Ask me any question about the material!"
    })
    st.session_state.first_load = False

# Display conversation history
for message in st.session_state.conversation:
    render_message(message["role"], message["content"], message.get("sources"))

# User input
if query := st.chat_input("Ask about a concept, formula, or problem...", disabled=st.session_state.processing):
    if not st.session_state.db_exists:
        st.error("The knowledge base has not been built yet. Please upload files and build it using the sidebar.", icon="📚")
    else:
        # Append user message and render immediately
        st.session_state.conversation.append({"role": "user", "content": query})
        st.session_state.chat_history.append({"role": "user", "content": query})
        render_message("user", query)

        st.session_state.processing = True
        with st.spinner("Thinking..."):
            try:
                # Get the full response
                answer, sources, pt = process_query(query, st.session_state.chat_history)
                
                # Create the assistant message dictionary
                assistant_message = {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                }
                
                # Append to conversation history and memory
                st.session_state.conversation.append(assistant_message)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

                # Re-render the new message from the assistant
                render_message(**assistant_message)
                st.caption(f"⏱️ Response generated in {pt:.2f} seconds.")

            except Exception as e:
                error_message = f"A critical error occurred: {e}"
                st.error(error_message)
                st.session_state.conversation.append({"role": "assistant", "content": f"⚠️ I encountered a problem: {error_message}"})
            
            finally:
                st.session_state.processing = False
                gc.collect()
                st.rerun() # Rerun to properly handle state changes
