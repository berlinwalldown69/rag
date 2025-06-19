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

# --- Removed Hardcoded Paths ---
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# POPPLER_PATH = r'C:\Users\vikra\RAG CFA\poppler\bin'
# --- End of Removed Hardcoded Paths ---


# Import PDF conversion library
try:
    from pdf2image import convert_from_path, pdfinfo_from_path
    PDF2IMAGE_AVAILABLE = True
    # For pdf2image, if poppler is not automatically found, you might need to set it
    # For Linux/Cloud deployments where poppler-utils is installed, it's often in /usr/bin
    # You might need to set an environment variable or find the path dynamically if it's not in default PATH
    # Example for a Docker environment (often not needed if installed via apt-get):
    # POPPLER_PATH_DEPLOYMENT = os.environ.get("POPPLER_PATH", None) # Set this env var if needed
    # if POPPLER_PATH_DEPLOYMENT and os.path.isdir(POPPLER_PATH_DEPLOYMENT):
    #     POPPLER_PATH = POPPLER_PATH_DEPLOYMENT
    # else:
    #     POPPLER_PATH = None # Rely on default system path
    # NOTE: For Streamlit Community Cloud with packages.txt, it should often find it automatically.
    # For Docker, the Dockerfile ensures it's in PATH.
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    st.warning("pdf2image is not installed. Please install it to process PDF files.")

# Determine POPPLER_PATH dynamically if needed, or rely on system PATH
# In most containerized or cloud environments, if poppler-utils is installed,
# pdf2image will find it without explicitly setting POPPLER_PATH.
# Keep this line if you still want to explicitly set it (e.g., if it's in a non-standard location)
# Otherwise, for simpler deployments, remove it and let pdf2image find it.
# POPPLER_PATH = os.environ.get("POPPLER_PATH") # Example: Set this as an environment variable during deployment


# --- Database for Caching ---
# The SQLite database will be ephemeral in most stateless cloud deployments.
# For persistent caching, consider cloud storage (S3, GCS) or a remote database.
def init_db():
    conn = sqlite3.connect("document_cache.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            file_hash TEXT PRIMARY KEY,
            documents TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def get_cached_documents(file_hash: str) -> Optional[List[Document]]:
    conn = sqlite3.connect("document_cache.db")
    c = conn.cursor()
    c.execute("SELECT documents FROM documents WHERE file_hash = ?", (file_hash,))
    result = c.fetchone()
    conn.close()
    if result:
        docs_json = json.loads(result[0])
        return [Document(**doc_data) for doc_data in docs_json]
    return None

def set_cached_documents(file_hash: str, documents: List[Document]):
    docs_json = json.dumps([doc.dict() for doc in documents])
    conn = sqlite3.connect("document_cache.db")
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO documents (file_hash, documents) VALUES (?, ?)", (file_hash, docs_json))
    conn.commit()
    conn.close()

def compute_file_hash(file_content: bytes) -> str:
    return hashlib.md5(file_content).hexdigest()

# --- Custom LLM using Google Generative AI ---
class CustomLLM(LLM):
    model_name: str = "gemini-1.5-flash"
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        genai.configure(api_key=self.google_api_key)
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        return response.text

    @property
    def _llm_type(self) -> str:
        return "custom"

# --- File Processing ---
@st.cache_data(show_spinner=False)
def process_uploaded_files(uploaded_files: List[Any]) -> List[Document]:
    all_docs = []
    with st.spinner("Processing documents... This may take a moment."):
        for file in uploaded_files:
            file_content = file.getvalue()
            file_hash = compute_file_hash(file_content)
            
            cached_docs = get_cached_documents(file_hash)
            if cached_docs:
                all_docs.extend(cached_docs)
                st.info(f"📄 Found cached version of '{file.name}'.")
                continue

            st.info(f"📄 Processing '{file.name}'...")
            docs = []
            tmp_file_path = None # Initialize to None
            try:
                # Use a more secure temporary file creation
                fd, tmp_file_path = tempfile.mkstemp(suffix=os.path.splitext(file.name)[1])
                with os.fdopen(fd, 'wb') as tmp_file:
                    tmp_file.write(file_content)

                extension = os.path.splitext(file.name)[1].lower()

                if extension == ".csv":
                    loader = CSVLoader(tmp_file_path)
                    docs.extend(loader.load())
                elif extension in [".xls", ".xlsx"]:
                    loader = UnstructuredExcelLoader(tmp_file_path, mode="elements")
                    docs.extend(loader.load())
                elif extension in [".ppt", ".pptx"]:
                    loader = UnstructuredPowerPointLoader(tmp_file_path, mode="elements")
                    docs.extend(loader.load())
                elif extension == ".pdf":
                    if PDF2IMAGE_AVAILABLE:
                        try:
                            # Rely on pdf2image finding poppler in system PATH or by default.
                            # Removed POPPLER_PATH argument, assume it's in system PATH.
                            info = pdfinfo_from_path(tmp_file_path) 
                            num_pages = info['Pages']
                            
                            for page_num in range(1, num_pages + 1):
                                images = convert_from_path(
                                    tmp_file_path, 
                                    first_page=page_num, 
                                    last_page=page_num
                                )
                                # Ensure pytesseract finds Tesseract executable (usually in system PATH)
                                text = pytesseract.image_to_string(images[0])
                                docs.append(Document(page_content=text, metadata={"source": file.name, "page": page_num}))
                        except Exception as e:
                            st.error(f"Error processing PDF '{file.name}': {e}")
                    else:
                        st.warning("Cannot process PDF files as pdf2image is not available.")
                elif extension in [".png", ".jpg", ".jpeg"]:
                    image = Image.open(tmp_file_path)
                    text = pytesseract.image_to_string(image)
                    docs.append(Document(page_content=text, metadata={"source": file.name}))
                else:
                    st.warning(f"Unsupported file type: {extension}")

                if docs:
                    set_cached_documents(file_hash, docs)
                    all_docs.extend(docs)

            except Exception as e:
                st.error(f"An error occurred while processing {file.name}: {e}")
            finally:
                if tmp_file_path and os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)

    return all_docs

# --- Vector Store and QA Chain ---
@st.cache_resource
def create_vector_store(_documents: List[Document]):
    if not _documents:
        return None
    with st.spinner("Creating vector store..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(_documents)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        try:
            vector_store = FAISS.from_documents(splits, embeddings)
            return vector_store
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            return None

@st.cache_data
def get_qa_chain(_vector_store, google_api_key: str):
    if _vector_store is None:
        return None
    llm = CustomLLM(google_api_key=google_api_key)
    retriever = _vector_store.as_retriever(search_kwargs={'k': 3})
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# --- Main App ---
def main():
    st.set_page_config(page_title="Document Q&A with Gemini", page_icon="🤖", layout="wide")
    st.title("📄 Document Q&A with Gemini Pro")
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    with st.sidebar:
        st.header("⚙️ Setup")
        
        google_api_key = st.text_input("Enter your Google API Key:", type="password")
        if not google_api_key:
            st.warning("Please enter your Google API Key to proceed.")
            st.stop()

        uploaded_files = st.file_uploader(
            "Upload your documents (PDF, CSV, XLS, PPT, PNG, JPG)", 
            type=["pdf", "csv", "xls", "xlsx", "ppt", "pptx", "png", "jpg", "jpeg"],
            accept_multiple_files=True
        )

        if st.button("Process Documents"):
            if uploaded_files:
                st.session_state.documents = process_uploaded_files(uploaded_files)
                st.session_state.vector_store = create_vector_store(st.session_state.documents)
                if st.session_state.vector_store:
                    st.success(f"{len(st.session_state.documents)} document pages processed successfully!")
                else:
                    st.error("Failed to create vector store. Please check the document formats and content.")
            else:
                st.warning("Please upload at least one document.")
    
    if 'vector_store' in st.session_state and st.session_state.vector_store:
        st.info("✅ Documents processed. You can now ask questions.")
        
        # Display conversation history
        for author, text in st.session_state.conversation:
            with st.chat_message(author):
                st.markdown(text)

        # User input
        query = st.chat_input("Ask a question about your documents...")

        if query:
            st.session_state.conversation.append(("user", query))
            with st.chat_message("user"):
                st.markdown(query)
            
            with st.spinner("🤖 Thinking..."):
                try:
                    qa_chain = get_qa_chain(st.session_state.vector_store, google_api_key)
                    if qa_chain:
                        result = qa_chain({"query": query})
                        response = result['result']
                        source_docs = result['source_documents']
                        
                        st.session_state.conversation.append(("assistant", response))
                        
                        with st.chat_message("assistant"):
                            st.markdown(response)
                            if source_docs:
                                with st.expander(f"📚 Sources ({len(source_docs)} found)"):
                                    for i, doc in enumerate(source_docs, 1):
                                        source = doc.metadata.get('source', 'Unknown')
                                        page = doc.metadata.get('page', 'N/A')
                                        
                                        st.markdown(f"**Source {i}:** `{os.path.basename(source)}` | **Page:** `{page}`")
                                        
                                        content_preview = doc.page_content[:300]
                                        if len(doc.page_content) > 300:
                                            content_preview += "..."
                                        st.markdown(f"> {content_preview}")
                                        
                                        if i < len(source_docs):
                                            st.markdown("---")
                    else:
                        st.error("QA chain is not available.")
                
                except Exception as e:
                    error_message = f"❌ Error generating response: {str(e)[:200]}..."
                    st.error(error_message)
                    st.session_state.conversation.append(("assistant", error_message))

    else:
        st.info("👆 Please upload and process your documents using the sidebar to begin asking questions.")

if __name__ == "__main__":
    main()