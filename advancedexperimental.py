import os
import glob
import sys
from typing import Optional, List, Any
from pathlib import Path
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
    # For langchain >= 0.1.0
    from langchain.chains import RetrievalQA
except ImportError:
    # For older versions
    from langchain.chains.retrieval_qa import RetrievalQA
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from PIL import Image
import pytesseract
import sqlite3
import google.generativeai as genai
from pydantic import Field

# Configure Tesseract path
# Configure Tesseract with type-safe approach
try:
    # Try to use Tesseract from system PATH
    pytesseract.get_tesseract_version()
    tesseract_cmd = "tesseract"  # Use literal for type checking
except pytesseract.TesseractNotFoundError:
    # Fallback to explicit path with type assertion
    TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(TESSERACT_PATH):
        os.environ["TESSERACT_PREFIX"] = os.path.dirname(TESSERACT_PATH)
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH  # type: ignore
    else:
        raise EnvironmentError("Tesseract not found in PATH or at default location")

# Import the new library to convert PDF to images
from pdf2image import convert_from_path

def check_dependencies():
    """Check if required dependencies are installed and configured."""
    
    # Check and configure Tesseract
    tesseract_installed = False
    try:
        # First try with default PATH
        pytesseract.get_tesseract_version()
        tesseract_installed = True
    except pytesseract.TesseractNotFoundError:
        # Try common Tesseract installation paths on Windows
        common_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        ]
        for path in common_paths:
            if os.path.exists(path):
                print(f"Found Tesseract at: {path}")
                pytesseract.pytesseract.tesseract_cmd = path  # type: ignore
                try:
                    pytesseract.get_tesseract_version()
                    tesseract_installed = True
                    break
                except:
                    continue

    if not tesseract_installed:
        print("Error: Tesseract is not installed or not properly configured.")
        print("\nTo fix this:")
        print("1. Download Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. Run the installer (make sure to check 'Add to PATH' during installation)")
        print("3. Expected installation paths:")
        print("   - C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
        print("   - C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe")
        print("\nIf already installed, please ensure Tesseract is in your PATH or")
        print("manually set the path in the code by adding this line at the top of the script:")
        print('pytesseract.pytesseract.tesseract_cmd = r"C:\\Path\\To\\Tesseract\\tesseract.exe"')
        return False
    
    # Check Poppler for Windows
    if sys.platform == "win32":
        poppler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "poppler", "bin")
        if not os.path.exists(poppler_path):
            print("\nError: Poppler is not found in the expected location.")
            print("Please download Poppler for Windows from: https://github.com/oschwartz10612/poppler-windows/releases/")
            print("Extract it and place the 'poppler' folder in the same directory as this script.")
            print(f"Expected path: {poppler_path}")
            return False
    
    print("All dependencies are properly configured!")
    return True

# --- NEW FUNCTION ---
# Custom loader for PDF files using OCR
def load_pdf_with_ocr(file_path: str) -> List[Document]:
    """
    Loads a PDF, converts each page to an image, and performs OCR to extract text.
    """
    documents = []
    try:
        # Set up Poppler path for Windows
        poppler_path = None
        if sys.platform == "win32":
            poppler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "poppler", "bin")
            if not os.path.exists(poppler_path):
                raise Exception("Poppler not found. Please check the dependency installation instructions.")
        
        # Convert PDF to images with explicit poppler path on Windows
        print(f"Converting PDF to images... This may take a while for large files.")
        if poppler_path and os.path.exists(poppler_path):
            images = convert_from_path(file_path, poppler_path=poppler_path)
        else:
            images = convert_from_path(file_path)
        
        print(f"Processing {len(images)} pages from {os.path.basename(file_path)} with OCR...")

        for i, image in enumerate(images):
            print(f"Processing page {i+1}/{len(images)}...")
            try:
                # Extract text and ensure it's a string
                extracted_text = pytesseract.image_to_string(image)
                text = str(extracted_text) if extracted_text else ""
                
                if text.strip():  # Only add pages with actual text
                    documents.append(
                        Document(
                            page_content=str(text),
                            metadata={
                                "source": str(file_path),
                                "page": i + 1
                            }
                        )
                    )
            except Exception as e:
                print(f"Error processing page {i+1}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"\nError processing PDF file {file_path}: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Verify Tesseract installation:")
        print(f"   Current Tesseract path: {pytesseract.pytesseract.tesseract_cmd}")
        print("2. Ensure Poppler is installed and in the correct location")
        print("3. Check if the PDF file is not corrupted")
        return []
        
    return documents

# Custom loader for .db files (SQLite assumed)
def load_from_db(file_path):
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    documents = []
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        for row in rows:
            text = ' '.join([str(item) for item in row])
            documents.append(Document(page_content=text, metadata={"source": file_path, "table": table_name}))
    conn.close()
    return documents

# Custom loader for image files using OCR
def load_from_image(file_path: str) -> List[Document]:
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        if isinstance(text, (bytes, bytearray)):
            text = text.decode('utf-8')
        return [Document(page_content=str(text), metadata={"source": str(file_path)})]
    except Exception as e:
        print(f"Error processing image file {file_path}: {e}")
        return []

# --- MODIFIED FUNCTION ---
# Function to load documents based on file extension
def load_documents_from_file(file_path):
    # Use the new OCR loader for PDF files
    if file_path.endswith('.pdf'):
        # Using the new OCR-based loader instead of PDFPlumberLoader
        return load_pdf_with_ocr(file_path)
    elif file_path.endswith('.csv'):
        loader = CSVLoader(file_path)
        return loader.load()
    elif file_path.endswith('.xlsx'):
        loader = UnstructuredExcelLoader(file_path)
        return loader.load()
    elif file_path.endswith('.pptx'):
        loader = UnstructuredPowerPointLoader(file_path)
        return loader.load()
    elif file_path.endswith('.db'):
        return load_from_db(file_path)
    elif file_path.endswith(('.jpg', '.png', '.jpeg')):
        return load_from_image(file_path)
    else:
        return []

# Custom LLM class for Google Gemini
class GeminiLLM(LLM):
    api_key: str = Field(..., description="Google API key for Gemini")

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return str(response.text)

    @property
    def _llm_type(self) -> str:
        return "gemini"

# Main function to set up and run the RAG model
def main():
    # Check dependencies first
    if not check_dependencies():
        print("\nPlease install the required dependencies and try again.")
        return

    # Prompt user for inputs
    api_key = input("Please enter your Google API key: ")
    path_input = input("Please enter the path to a file or directory: ")

    # Load documents from the path
    documents = []
    
    if os.path.isfile(path_input):
        # If the input is a file, process just that file
        print(f"Loading from file: {path_input}...")
        docs = load_documents_from_file(path_input)
        if docs:
            print(f"Loaded {len(docs)} documents from {os.path.basename(path_input)}")
        else:
            print(f"Warning: No documents were loaded from {os.path.basename(path_input)}")
        documents.extend(docs)
    else:
        # If the input is a directory, process all files in it
        file_list = glob.glob(os.path.join(path_input, '*'))
        if not file_list:
            print(f"No files found in directory: {path_input}")
            return

        for file_path in file_list:
            print(f"Loading from {file_path}...")
            docs = load_documents_from_file(file_path)
            if docs:
                print(f"Loaded {len(docs)} documents from {os.path.basename(file_path)}")
            else:
                print(f"Warning: No documents were loaded from {os.path.basename(file_path)}")
            documents.extend(docs)

    if not documents:
        print("No documents loaded. Exiting.")
        return

    # Debug: Print the first few documents' content length and preview
    for i, doc in enumerate(documents[:5]):
        print(f"\n--- Preview of Document {i+1} ---")
        print(f"Content length: {len(doc.page_content)}")
        print(f"Content preview: {repr(doc.page_content[:200])}")
        print("--------------------------")

    # Filter out empty or whitespace-only documents
    documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
    print(f"\nTotal documents with non-empty content after filtering: {len(documents)}")
    if not documents:
        print("All loaded documents are empty. Exiting.")
        return

    # Split documents into chunks for efficient retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    print(f"Total splits created: {len(splits)}")
    if not splits:
        print("No splits created from documents. Exiting.")
        return

    # Create embeddings and index them with FAISS
    print("Creating embeddings and vector store... This may take a moment.")
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    # Initialize the Gemini LLM
    llm = GeminiLLM(api_key=api_key)

    # Set up the RAG model with RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # Interactive querying loop
    print("\nRAG model is ready. You can now ask questions based on your documents.")
    while True:
        query = input("Enter your query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        response = qa_chain.run(query)
        print("Response:", response)

if __name__ == "__main__":
    main()
