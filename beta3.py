import os
import gc
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import argparse
from langchain.prompts import load_prompt
from langchain.chains import LLMChain
from langchain.agents import Tool, AgentExecutor
from langchain.agents import initialize_agent
import time

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Configuration
DB_PATH = "./cfa_chroma_db"
PDF_DIR = "/Users/vikramleo/Downloads/RAG/cfa_books/"
RAG_TOPK = 5
ADD_BATCH_SIZE = 1
MAX_CHARACTERS = 4000

def enhanced_query_engine(question, context):
    """Multi-stage reasoning chain with reflection capabilities"""
    
    # Stage 1: Generate step-by-step reasoning
    step_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a CFA expert assistant. Analyze the question and contextual CFA materials to develop a step-by-step reasoning path."),
        ("human", """
**Context:**
{context}

**Question:** {question}

Develop a clear reasoning path to answer the above CFA question:
1. Identify the specific CFA domain and concepts involved
2. Break down complex terms or calculations
3. Outline required logical steps to reach the solution
4. Highlight any assumptions based on CFA principles        
        """)
    ])
    
    # Stage 2: Critique and improve reasoning
    critique_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a CFA Level 3 grader. Identify gaps in the reasoning and improve it."),
        ("human", """
**Original Reasoning:** 
{reasoning}

**Critique the reasoning based on CFA standards:**
- Identify logical fallacies or gaps
- Validate calculation approaches
- Ensure compliance with CFA curriculum
- Suggest improvements
        """)
    ])
    
    # Stage 3: Produce final answer
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional CFA charterholder. Synthesize the reasoning and critique to provide the final answer."),
        ("human", """
**Question:** {question}
**Reasoning:** {reasoning}
**Critique:** {critique}

**Deliver your response:**
- Answer with professional precision
- Format complex calculations clearly
- Reference CFA curriculum where appropriate
- Provide theory + practical application        
        """)
    ])
    
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=API_KEY,
        temperature=0.35,
        max_output_tokens=4096
    )
    
    # Execute multi-step chain
    chain1 = step_prompt | model | StrOutputParser()
    reasoning = chain1.invoke({"context": context, "question": question})
    
    chain2 = critique_prompt | model | StrOutputParser()
    critique = chain2.invoke({"reasoning": reasoning})
    
    chain3 = answer_prompt | model | StrOutputParser()
    final_answer = chain3.invoke({
        "question": question, 
        "reasoning": reasoning, 
        "critique": critique
    })
    
    return final_answer, reasoning, critique

def process_pdfs():
    """Process PDFs incrementally and update ChromaDB."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=API_KEY
    )
    
    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name="cfa_knowledge"
    )
    
    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(PDF_DIR, filename)
            print(f"Processing {filename}...")
            
            loader = PyPDFLoader(filepath)
            data = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=150
            )
            all_splits = text_splitter.split_documents(data)
            
            # Add documents to ChromaDB in batches
            for i in range(0, len(all_splits), ADD_BATCH_SIZE):
                batch = all_splits[i:i + ADD_BATCH_SIZE]
                vector_db.add_documents(batch)
                
            print(f"Finished processing {filename}. Total documents in DB: {vector_db._collection.count()}")
            
            gc.collect() # Clean up memory
            
    vector_db.persist()
    print("\n📚 PDF processing complete! Knowledge base built.")

def query_cfa(question):
    """Enhanced RAG pipeline with multi-step reasoning"""
    try:
        # Initialize ChromaDB
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=API_KEY
        )
        
        vector_db = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings,
            collection_name="cfa_knowledge"
        )
        
        # Retrieve relevant context
        retriever = vector_db.as_retriever(search_kwargs={"k": RAG_TOPK})
        docs = retriever.get_relevant_documents(question)
        context = "\n\n---\n\n".join([
            f"Source: {os.path.basename(doc.metadata['source'])}\nContent: {doc.page_content}" 
            for doc in docs
        ])
        
        # Generate multi-step reasoned answer
        answer, reasoning, critique = enhanced_query_engine(question, context)
        
        # Add source information
        sources = "\n".join(set(
            f"• {os.path.basename(doc.metadata['source'])}" 
            for doc in docs
        ))
        
        return f"{answer}\n\n🧠 Reasoning Path:\n{reasoning}\n\n🛠 Critique:\n{critique}\n\n📚 Sources:\n{sources}"
        
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
    finally:
        gc.collect()

def interactive_query():
    """Enhanced interactive interface with timing"""
    print("\n💬 CFA Expert Reasoning Engine")
    print("Type 'exit' to quit\n")
    
    while True:
        try:
            question = input("\n📝 Your CFA question: ")
            if question.lower() in ["exit", "quit"]:
                break
                
            start_time = time.time()
            response = query_cfa(question)
            duration = time.time() - start_time
            
            print(f"\n🔍 Response [processed in {duration:.1f}s]:\n")
            print(response)
            print("\n" + "═" * 80)
            
        except Exception as e:
            print(f"⚠️ Error: {str(e)}")
            
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CFA Reasoning Engine")
    parser.add_argument('--process', action='store_true', help='Process PDFs')
    parser.add_argument('--query', action='store_true', help='Interactive Q/A')
    parser.add_argument('--ask', type=str, help='Ask a single question')
    
    args = parser.parse_args()
    
    if args.process:
        process_pdfs()
    elif args.query:
        interactive_query()
    elif args.ask:
        response = query_cfa(args.ask)
        print(f"\n📝 Your question: {args.ask}")
        print(f"\n💡 Expert Answer:\n{response}")
    else:
        print("Usage:\n  --process  Build knowledge base\n  --query    Interactive mode\n  --ask      Ask single question")
