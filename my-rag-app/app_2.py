# -------------------------------------------
# My RAG App - Happy Learning!
# -------------------------------------------

import streamlit as st
import os
import shutil
import google.generativeai as genai
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. CORE AI LOGIC (Your previous work)
# ==========================================
def find_models(api_key):
    genai.configure(api_key=api_key)
    embed, chat = None, None
    for m in genai.list_models():
        if not embed and 'embedContent' in m.supported_generation_methods:
            embed = m.name
        if not chat and 'generateContent' in m.supported_generation_methods:
            chat = m.name
    return embed, chat

# ==========================================
# 2. STREAMLIT GUI CONFIGURATION
# ==========================================
st.set_page_config(page_title="RAG Research Bot", page_icon="🤖")
st.title("📄 PDF Intelligence System")

# Initialize "Database" for Chat History in memory
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# Sidebar for Setup
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter Google API Key", type="password")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if st.button("Build Knowledge Base"):
        if not api_key or not uploaded_file:
            st.error("Please provide both API Key and a PDF file.")
        else:
            with st.spinner("Processing PDF (OCR + Embedding)..."):
                # Save uploaded file locally
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Setup models
                embed_name, chat_name = find_models(api_key)
                st.session_state.chat_model = chat_name
                
                # Load and Chunk
                loader = PyMuPDFLoader("temp.pdf", extract_images=True, images_parser=RapidOCRBlobParser())
                data = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                chunks = text_splitter.split_documents(data)
                
                # Embed into Vector DB
                embeddings = GoogleGenerativeAIEmbeddings(
                    model=embed_name, google_api_key=api_key, task_type="retrieval_document"
                )
                
                if os.path.exists("./chroma_db"):
                    shutil.rmtree("./chroma_db")
                
                st.session_state.vector_db = Chroma.from_documents(
                    documents=chunks, embedding=embeddings, persist_directory="./chroma_db"
                )
                st.success("System Ready!")

# ==========================================
# 3. CHAT INTERFACE
# ==========================================
# Display chat history from session database
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about your document..."):
    if not st.session_state.vector_db:
        st.warning("Please upload and process a PDF first!")
    else:
        # 1. Add user message to UI database
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. RAG Execution
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Setup Retriever
                retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})
                llm = ChatGoogleGenerativeAI(model=st.session_state.chat_model, google_api_key=api_key)
                
                template = "Answer based only on context: {context}\nQuestion: {question}"
                prompt_template = ChatPromptTemplate.from_template(template)
                
                chain = (
                    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
                     "question": RunnablePassthrough()}
                    | prompt_template | llm | StrOutputParser()
                )
                
                response = chain.invoke(prompt)
                st.markdown(response)
                
                # 3. Add AI response to UI database
                st.session_state.messages.append({"role": "assistant", "content": response})