import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

def test_environment():
    print("\nTesting Environment Setup...")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    
    # Test environment variables
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("❌ GEMINI_API_KEY not found in environment")
    print("✅ API key loaded successfully")

def test_embeddings():
    print("\nTesting Embeddings...")
    api_key = os.getenv("GEMINI_API_KEY")
    
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
        task_type="retrieval_document"
    )
    
    # Test with a sample text
    test_text = "CFA Level 1 Portfolio Management"
    result = embeddings.embed_query(test_text)
    if not isinstance(result, list) or len(result) == 0:
        raise ValueError("❌ Embedding generation failed")
    print(f"✅ Embeddings working (vector size: {len(result)})")
    return embeddings

def test_vector_db(embeddings):
    print("\nTesting Vector Database...")
    
    # Ensure directories exist
    db_path = "./vector_db"
    Path(db_path).mkdir(parents=True, exist_ok=True)
    
    try:
        vector_db = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings,
            collection_name="cfa_knowledge"
        )
        doc_count = vector_db._collection.count()
        print(f"✅ Vector database initialized (documents: {doc_count})")
        return vector_db
    except Exception as e:
        print(f"❌ Vector database error: {str(e)}")
        raise

def main():
    try:
        print("🔄 Starting component tests...")
        test_environment()
        embeddings = test_embeddings()
        vector_db = test_vector_db(embeddings)
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
