import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

def test_setup():
    print("Testing setup...")
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    print("✅ API key loaded")

    # Test embeddings
    print("Testing embeddings initialization...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
        task_type="retrieval_document"
    )
    test_embedding = embeddings.embed_query("test")
    if not test_embedding:
        raise ValueError("Embedding test failed")
    print("✅ Embeddings working")

    # Test vector database
    print("Testing vector database initialization...")
    db_path = "./vector_db"
    os.makedirs(db_path, exist_ok=True)
    
    vector_db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name="cfa_knowledge"
    )
    print(f"✅ Vector database initialized (document count: {vector_db._collection.count()})")
    return True

if __name__ == "__main__":
    try:
        test_setup()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
