__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="PDF Intelligence System", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    .stTextInput>div>div>input { background-color: #262730; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SECRET MANAGEMENT ---
# This part automatically grabs the key from the "Vault"
if "GOOGLE_API_KEY" in st.secrets:
    DEFAULT_API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    DEFAULT_API_KEY = ""

# --- 3. LOGIC FUNCTIONS ---
def find_models(api_key):
    try:
        genai.configure(api_key=api_key)
        embed, chat = None, None
        for m in genai.list_models():
            if not embed and 'embedContent' in m.supported_generation_methods: embed = m.name
            if not chat and 'generateContent' in m.supported_generation_methods: chat = m.name
        return embed, chat
    except Exception: return None, None

# --- 4. SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "vector_db" not in st.session_state: st.session_state.vector_db = None

# --- 5. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("Control Panel")
    
    # User only sees this if the Secret is missing
    api_key = st.text_input("Gemini API Key", value=DEFAULT_API_KEY, type="password")
    
    
    uploaded_files = st.file_uploader("Source Technical Documents (PDFs)", type="pdf", accept_multiple_files=True)
    
    if st.button("🚀 Initialize System"):
        if api_key and uploaded_files:
            with st.spinner(f"🔧 Processing {len(uploaded_files)} files..."):
                all_chunks = [] # Master list for all PDF data
                
                for uploaded_file in uploaded_files:
                    # Save each file temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Load and chunk this specific file
                    loader = PyMuPDFLoader(temp_path, extract_images=True, images_parser=RapidOCRBlobParser())
                    data = loader.load()
                    
                    # We add the filename to metadata so the AI knows which file is which
                    for doc in data:
                        doc.metadata["source_file"] = uploaded_file.name
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    chunks = text_splitter.split_documents(data)
                    all_chunks.extend(chunks) # Add to the master list
                    
                    os.remove(temp_path) # Clean up temp file

                # Build ONE database for all chunks
                embed_name, chat_name = find_models(api_key)
                st.session_state.chat_model = chat_name
                embeddings = GoogleGenerativeAIEmbeddings(model=embed_name, google_api_key=api_key)
                
                if os.path.exists("/tmp/chroma_db"): shutil.rmtree("/tmp/chroma_db")
                st.session_state.vector_db = Chroma.from_documents(
                    documents=all_chunks, 
                    embedding=embeddings, 
                    persist_directory="/tmp/chroma_db"
                )
                st.success(f"✅ System Online with {len(uploaded_files)} documents!")

# --- 6. MAIN CHAT INTERFACE ---
st.subheader("🤖 AI Research Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

if prompt := st.chat_input("Query the knowledge base..."):
    if not st.session_state.vector_db:
        st.error("System Offline: Please initialize in the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})
            source_docs = retriever.invoke(prompt)
            
            llm = ChatGoogleGenerativeAI(model=st.session_state.chat_model, google_api_key=api_key, temperature=0.1)
            template = """Answer based only on context:\n{context}\n\nQuestion: {question}"""
            chain = ({"context": lambda x: "\n\n".join(d.page_content for d in source_docs), "question": RunnablePassthrough()} 
                     | ChatPromptTemplate.from_template(template) | llm | StrOutputParser())
            
            response = chain.invoke(prompt)
            st.markdown(response)
            
            with st.expander("🔍 View Source Material"):
                for i, doc in enumerate(source_docs):
                    st.info(f"**Source {i+1}:** {doc.page_content[:300]}...")

            st.session_state.messages.append({"role": "assistant", "content": response})