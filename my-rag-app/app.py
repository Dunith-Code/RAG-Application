import os
import glob
import shutil
import google.generativeai as genai
from langchain_community.document_loaders import PyMuPDFLoaqder
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. CONFIGURATION
# ==========================================
MY_API_KEY = "API_KEY"   # <-- Replace with your Google AI Studio API key

def find_embedding_model():
    """Find the first available embedding model for the API key."""
    genai.configure(api_key=MY_API_KEY)
    for m in genai.list_models():
        if 'embedContent' in m.supported_generation_methods:
            return m.name
    return None

def find_chat_model():
    """Find the first available model that supports generateContent."""
    genai.configure(api_key=MY_API_KEY)
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            return m.name
    return None

def run_rag_app():
    # Find the first PDF file in the current folder
    pdf_files = glob.glob("*.pdf")
    if not pdf_files:
        print("Error: No PDF found.")
        return
    
    target_pdf = pdf_files[0]
    print(f"Loading: {target_pdf}")

    # Clean any existing local Chroma database
    if os.path.exists("./chroma_db_final"):
        shutil.rmtree("./chroma_db_final", ignore_errors=True)

    # 2. LOAD & CHUNK
    loader = PyMuPDFLoader(target_pdf, extract_images=True, images_parser=RapidOCRBlobParser())
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    print(f"Ready with {len(chunks)} chunks.")

    # 3. AUTO-DETECT EMBEDDINGS
    embed_model_name = find_embedding_model()
    if not embed_model_name:
        print("Critical Error: Could not find any valid embedding models for your key.")
        return
        
    print(f"Using Embedding Model: {embed_model_name}")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model=embed_model_name, 
        google_api_key=MY_API_KEY,
        task_type="retrieval_document"
    )
    
    print("Creating vector database...")
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory="./chroma_db_final"
    )

    # 4. AUTO-DETECT CHAT MODEL
    chat_model_name = find_chat_model()
    if not chat_model_name:
        print("Critical Error: No chat model found for your API key.")
        return

    print(f"Using Chat Model: {chat_model_name}")

    llm = ChatGoogleGenerativeAI(
        model=chat_model_name,
        google_api_key=MY_API_KEY,
        temperature=0.1
    )
    
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    template = """Answer based ONLY on context: {context}\nQuestion: {question}"""
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    print("\n✅ System Ready! Ask your question.")
    while True:
        query = input("\nUser: ")
        if query.lower() in ["exit", "quit"]:
            break
        print("Thinking...")
        try:
            print(f"AI: {rag_chain.invoke(query)}")
        except Exception as e:
            print(f"Error during Chat: {e}")

if __name__ == "__main__":
    run_rag_app()