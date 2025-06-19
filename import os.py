import os
import re
import PyPDF2
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load environment variables (create a .env file with GEMINI_API_KEY=your_key)
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

# 1. PDF Processing Function
def extract_pdf_text(pdf_path):
    """Extract clean text from PDF files"""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            # Basic cleaning
            cleaned_text = re.sub(r'\s+', ' ', page_text).strip()
            text += cleaned_text + "\n"
    return text

# 2. Text Chunking Strategy
def split_text_chunks(text, chunk_size=1000, overlap=100):
    """Split text into manageable chunks with overlap"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start = end - overlap  # Overlap chunks
    return chunks

# 3. Initialize ChromaDB with Gemini Embeddings
gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=API_KEY)
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(
    name="cfa_knowledge",
    embedding_function=gemini_ef
)

# 4. Process PDFs and Populate Vector Database
def process_pdfs(pdf_directory):
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    all_chunks = []
    metadata = []
    ids = []

    for i, filename in enumerate(pdf_files):
        print(f"Processing {filename}...")
        text = extract_pdf_text(os.path.join(pdf_directory, filename))
        chunks = split_text_chunks(text)
        
        for j, chunk in enumerate(chunks):
            chunk_id = f"{filename}_chunk{j}"
            ids.append(chunk_id)
            all_chunks.append(chunk)
            metadata.append({"source": filename})

    # Add to ChromaDB in batches
    for i in range(0, len(all_chunks), 100):  # Batch size 100
        batch_ids = ids[i:i+100]
        batch_chunks = all_chunks[i:i+100]
        batch_metadata = metadata[i:i+100]
        
        collection.add(
            documents=batch_chunks,
            metadatas=batch_metadata,
            ids=batch_ids
        )
    print(f"Processed {len(all_chunks)} chunks from {len(pdf_files)} files")

# 5. Query Processing System
def rag_query(question, max_results=5):
    # First retrieve relevant context
    results = collection.query(
        query_texts=[question],
        n_results=max_results
    )
    
    # Build context from top matches
    context = "\n\n".join(results['documents'][0])
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(
        f"Using ONLY the following CFA materials context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely and professionally without mentioning your knowledge cutoff date:"
    )
    return response.text

# ======= Usage Example =======
if __name__ == "__main__":
    # Step 1: Process PDFs (only need to run once)
    # IMPORTANT: Replace "path/to/your/cfa_books/" with the actual path to your PDF directory
    PDF_DIR = "/Users/vikramleo/Downloads/RAG/cfa_books/" 
    process_pdfs(PDF_DIR)
    
    # Step 2: Query your knowledge base
    question = "Explain the key differences between IRR and NPV in capital budgeting."
    answer = rag_query(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
