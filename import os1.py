import os
import re
import PyPDF2
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import argparse

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

# Configuration
DB_PATH = "./chroma_db"  # Persistent storage directory
COLLECTION_NAME = "cfa_knowledge"
PDF_DIR = "/Users/vikramleo/Downloads/RAG/cfa_books/"  # Update with your path

# ---- Core Functions ----
def extract_pdf_text(pdf_path):
    """Extract clean text from PDF files"""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            cleaned_text = re.sub(r'\s+', ' ', page_text).strip()
            text += cleaned_text + "\n"
    return text

def split_text_chunks(text, chunk_size=750, overlap=150):
    """Split text into manageable chunks with overlap"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        if end > len(words):
            end = len(words)
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def setup_vector_db():
    """Initialize ChromaDB with persistence and Gemini embeddings"""
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=API_KEY
    )
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=gemini_ef
    )

def process_pdfs():
    """Process and store PDF content in the vector database"""
    collection = setup_vector_db()
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    total_docs = 0
    
    for filename in pdf_files:
        full_path = os.path.join(PDF_DIR, filename)
        print(f"➡️ Processing {filename}...")
        
        try:
            text = extract_pdf_text(full_path)
            chunks = split_text_chunks(text)
            doc_ids = [f"{filename}-chunk-{i}" for i in range(len(chunks))]
            
            # Upload in batches
            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                batch_ids = doc_ids[i:i+batch_size]
                
                collection.add(
                    documents=batch,
                    ids=batch_ids,
                    metadatas=[{"source": filename}] * len(batch)
                )
            
            total_docs += len(chunks)
            print(f"✅ Added {len(chunks)} chunks from {filename}")
        except Exception as e:
            print(f"❌ Error processing {filename}: {str(e)}")
    
    print(f"\n🎉 Processing complete! Added {total_docs} total documents")
    return total_docs

def query_knowledge(question, max_results=7):
    """Query the vector database and generate response"""
    collection = setup_vector_db()
    
    # Retrieve relevant context
    results = collection.query(
        query_texts=[question],
        n_results=max_results
    )
    
    # Build context from top results
    context = "\n\n---\n\n".join([
        f"Source: {results['metadatas'][0][i]['source']}\nContent: {doc}" 
        for i, doc in enumerate(results['documents'][0])
    ])
    
    # Generate response using Gemini
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""You are a CFA expert assistant. Answer the question using ONLY the context below.
    
Context:
{context}

Question: {question}

Answer in a professional tone, citing sources where applicable. If information is missing, say "This isn't covered in the materials I have." 
"""
    response = model.generate_content(prompt)
    return response.text

def interactive_query():
    """Interactive query interface"""
    print("\n🔍 CFA Knowledge Query System")
    print("Type 'exit' to quit\n")
    
    while True:
        question = input("📝 Your question: ")
        if question.lower() in ['exit', 'quit']:
            break
        
        try:
            answer = query_knowledge(question)
            print(f"\n💡 Response:\n{answer}\n")
            print("-" * 50)
        except Exception as e:
            print(f"❌ Error: {str(e)}")

# ---- CLI Interface ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CFA Knowledge Base System')
    subparsers = parser.add_subparsers(dest='command')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process PDF files')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Run interactive query')
    
    # Single query command
    ask_parser = subparsers.add_parser('ask', help='Ask a single question')
    ask_parser.add_argument('question', type=str, help='Your question')
    
    args = parser.parse_args()
    
    if args.command == 'process':
        process_pdfs()
    elif args.command == 'query':
        interactive_query()
    elif args.command == 'ask':
        response = query_knowledge(args.question)
        print(f"\n💡 Question: {args.question}")
        print(f"\n💡 Response:\n{response}")
    else:
        parser.print_help()
