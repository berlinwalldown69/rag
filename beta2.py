import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import argparse
import gc  # Garbage collection for memory management

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Configuration
DB_PATH = "./cfa_chroma_db"
PDF_DIR = "/Users/vikramleo/Downloads/RAG/cfa_books/"
CHUNK_SIZE = 1000  # Reduced for low memory
CHUNK_OVERLAP = 100
ADD_BATCH_SIZE = 1  # Process one book at a time
MAX_CHARACTERS = 4000  # Lower for documents with complex formatting

def convert_to_markdown(text):
    """Convert text to simplified markdown for better parsing"""
    # Heading detection
    text = re.sub(r'(\n\s*[A-Z][A-Z0-9 \t]+)\n', r'\n# \1\n', text)
    # Bullet points
    text = re.sub(r'\n\s*•\s+', r'\n• ', text)
    return text

def process_pdfs():
    """Process PDFs incrementally to minimize memory usage"""
    print(f"\n📚 Building CFA Knowledge Base (low-memory mode)")
    
    # Initialize with low resource usage
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=API_KEY,
        task_type="retrieval_document"
    )
    
    # Initialize Chroma with consistent collection name
    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name="cfa_knowledge"
    )
    
    # Get list of PDFs but process one by one
    pdf_files = [
        os.path.join(PDF_DIR, f) 
        for f in os.listdir(PDF_DIR) 
        if f.lower().endswith('.pdf')
    ]
    
    documents_added = 0
    
    for pdf_path in pdf_files:
        try:
            print(f"\n📖 Processing: {os.path.basename(pdf_path)}")
            
            # Load with garbage collection
            loader = PyPDFLoader(pdf_path)
            text = ""
            for page_index, page in enumerate(loader.lazy_load()):
                text += page.page_content[:MAX_CHARACTERS] + "\n\n"
                
                # Process in chunks to minimize memory
                if page_index % 10 == 0:
                    # Convert to markdown before splitting
                    processed_text = convert_to_markdown(text)
                    
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP,
                        length_function=len
                    )
                    
                    chunks = splitter.split_text(processed_text)
                    vector_db.add_texts(
                        texts=chunks,
                        metadatas=[{"source": pdf_path} for _ in chunks]
                    )
                    documents_added += len(chunks)
                    
                    # Clear memory
                    text = ""
                    gc.collect()
                    print(f"💾 Saved {len(chunks)} chunks (total: {documents_added})")
                
            # Process any remaining text
            if text.strip():
                processed_text = convert_to_markdown(text)
                chunks = splitter.split_text(processed_text)
                vector_db.add_texts(
                    texts=chunks,
                    metadatas=[{"source": pdf_path} for _ in chunks]
                )
                documents_added += len(chunks)
                print(f"💾 Saved final {len(chunks)} chunks")
            
            # Explicit persistence to reduce memory footprint
            vector_db.persist()
            
        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {e}")
        finally:
            # Force clear memory after each book
            gc.collect()
    
    print(f"\n✅ Completed! Added {documents_added} documents")
    return documents_added

def query_cfa(question):
    """Query the knowledge base with resource-optimized pipeline"""
    try:
        # Initialize with minimal resources
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=API_KEY
        )
        
        vector_db = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings,
            collection_name="cfa_knowledge"
        )
        
        # Use smaller retrieval batch
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        
        # Configure lower-resource model
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=API_KEY,
            temperature=0.3,
            max_output_tokens=2048
        )
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        result = qa.invoke({"query": question})
        
        # Format response
        response = result['result']
        sources = "\n".join(set(
            f"• {os.path.basename(doc.metadata['source'])}" 
            for doc in result['source_documents']
        ))
        
        return f"{response}\n\n📚 Sources:\n{sources}"
        
    except Exception as e:
        return f"⚠️ Query failed: {str(e)}"

def interactive_query():
    """Memory-friendly query interface"""
    print("\n💬 CFA Query Mode (type 'exit' to quit)")
    while True:
        try:
            query = input("\n🧠 Your question: ")
            if query.lower() in ["exit", "quit"]:
                break
                
            response = query_cfa(query)
            print(f"\n💡 Response:\n\n{response}\n")
            print("─" * 80)
            
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"⚠️ Error: {str(e)}")
            
        # Force garbage collection after each query
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Low-memory CFA Knowledge System")
    parser.add_argument('--process', action='store_true', help='Process PDFs to create knowledge base')
    parser.add_argument('--query', action='store_true', help='Enter interactive query mode')
    parser.add_argument('--ask', type=str, help='Ask a single question')
    
    args = parser.parse_args()
    
    if args.process:
        process_pdfs()
    elif args.query:
        interactive_query()
    elif args.ask:
        response = query_cfa(args.ask)
        print(f"\n💬 Question: {args.ask}")
        print(f"\n💡 Response:\n{response}")
    else:
        print("No action specified. Use --process, --query, or --ask\n")
        parser.print_help()
