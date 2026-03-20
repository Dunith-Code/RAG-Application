# ===================================================================
# SQLITE WORKAROUND – MUST BE THE VERY FIRST LINES
# ===================================================================
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ===================================================================
# IMPORTS
# ===================================================================
import streamlit as st
import os
import shutil
import json
from datetime import datetime
import google.generativeai as genai
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS      # <-- FAISS instead of Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ===================================================================
# 1. SETTINGS & STYLING
# ===================================================================
st.set_page_config(page_title="PDF Intelligence System", page_icon="🛡️", layout="wide")

HISTORY_DIR = "chat_sessions"
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    .stTextInput>div>div>input { background-color: #262730; color: white; }
    </style>
    """, unsafe_allow_html=True)

# ===================================================================
# 2. SECRET MANAGEMENT (API key from Streamlit secrets)
# ===================================================================
if "GOOGLE_API_KEY" in st.secrets:
    SECRET_API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("🚨 CRITICAL ERROR: GOOGLE_API_KEY not found in Secrets Vault.")
    st.info("Please add the key to your Streamlit Cloud Secrets or local secrets.toml file.")
    st.stop()

# ===================================================================
# 3. HELPER FUNCTIONS
# ===================================================================
def find_models():
    """Find available embedding and chat models using the global secret."""
    try:
        genai.configure(api_key=SECRET_API_KEY)
        embed, chat = None, None
        for m in genai.list_models():
            if not embed and 'embedContent' in m.supported_generation_methods:
                embed = m.name
            if not chat and 'generateContent' in m.supported_generation_methods:
                chat = m.name
        return embed, chat
    except Exception as e:
        st.error(f"Error listing models: {e}")
        return None, None

def save_chat_to_file():
    if st.session_state.messages:
        first_msg = st.session_state.messages[0]["content"][:15].strip().replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{HISTORY_DIR}/Chat_{first_msg}_{timestamp}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)

def load_chat_from_file(filename):
    filepath = os.path.join(HISTORY_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        st.session_state.messages = json.load(f)

# ===================================================================
# 4. SESSION STATE
# ===================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "chat_model" not in st.session_state:
    st.session_state.chat_model = None

# ===================================================================
# 5. SIDEBAR – CONTROL PANEL
# ===================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("Control Panel")

    # Chat History section
    st.subheader("📜 Saved Histories")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ New Chat"):
            save_chat_to_file()
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🗑️ Clear"):
            st.session_state.messages = []
            st.rerun()

    history_files = os.listdir(HISTORY_DIR)
    if history_files:
        for f in sorted(history_files, reverse=True):
            if st.button(f"📁 {f[:20]}...", key=f):
                load_chat_from_file(f)
                st.rerun()
    else:
        st.info("No saved chats yet.")
    st.divider()

    # File upload and initialisation
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if st.button("🚀 Initialize System"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner(f"🔧 Processing {len(uploaded_files)} file(s)..."):
                try:
                    all_chunks = []
                    for uploaded_file in uploaded_files:
                        temp_path = f"temp_{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        loader = PyMuPDFLoader(temp_path, extract_images=True, images_parser=RapidOCRBlobParser())
                        data = loader.load()
                        for doc in data:
                            doc.metadata["source_file"] = uploaded_file.name

                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                        all_chunks.extend(text_splitter.split_documents(data))
                        os.remove(temp_path)

                    # Find models
                    embed_name, chat_name = find_models()
                    if not embed_name or not chat_name:
                        st.error("Authentication Error: Could not find suitable models. Check API key.")
                        st.stop()

                    st.session_state.chat_model = chat_name
                    embeddings = GoogleGenerativeAIEmbeddings(model=embed_name, google_api_key=SECRET_API_KEY)

                    # Create FAISS vector store (in‑memory, no SQLite)
                    st.session_state.vector_db = FAISS.from_documents(all_chunks, embeddings)

                    st.success(f"✅ System Online with {len(uploaded_files)} document(s)!")

                except Exception as e:
                    st.error(f"Initialization failed: {e}")
                    st.stop()

    if st.button("🗑️ Reset All"):
        if os.path.exists(HISTORY_DIR):
            shutil.rmtree(HISTORY_DIR)
            os.makedirs(HISTORY_DIR)
        st.session_state.messages = []
        st.session_state.vector_db = None
        st.session_state.chat_model = None
        st.rerun()

# ===================================================================
# 6. MAIN CHAT INTERFACE
# ===================================================================
st.subheader("🤖 PDF Intelligence System")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Query the knowledge base..."):
    if not st.session_state.vector_db:
        st.error("System Offline: Initialize in the sidebar first.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})
                source_docs = retriever.invoke(prompt)

                llm = ChatGoogleGenerativeAI(
                    model=st.session_state.chat_model,
                    google_api_key=SECRET_API_KEY,
                    temperature=0.1
                )

                template = "Answer based on context:\n{context}\n\nQuestion: {question}"
                chain = (
                    {
                        "context": lambda _: "\n\n".join(d.page_content for d in source_docs),
                        "question": RunnablePassthrough()
                    }
                    | ChatPromptTemplate.from_template(template)
                    | llm
                    | StrOutputParser()
                )

                try:
                    response = chain.invoke(prompt)
                    st.markdown(response)
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
                    response = "Sorry, I couldn't generate an answer."

                # Show sources
                with st.expander("🔍 Sources"):
                    for i, doc in enumerate(source_docs):
                        st.info(f"**Source {i+1}** ({doc.metadata.get('source_file', 'unknown')}):\n{doc.page_content[:200]}...")

                st.session_state.messages.append({"role": "assistant", "content": response})