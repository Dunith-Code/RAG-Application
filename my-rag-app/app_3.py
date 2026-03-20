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
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ===================================================================
# 1. SETTINGS & STYLING
# ===================================================================
st.set_page_config(page_title="PDF Intelligence System", page_icon="🛡️", layout="wide")

# Directory for chat history
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
# 2. SECRET MANAGEMENT
# ===================================================================
if "GOOGLE_API_KEY" in st.secrets:
    DEFAULT_API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    DEFAULT_API_KEY = ""

# ===================================================================
# 3. HELPER FUNCTIONS
# ===================================================================
def find_models(api_key):
    """Find available embedding and chat models."""
    try:
        genai.configure(api_key=api_key)
        embed, chat = None, None
        for m in genai.list_models():
            if not embed and 'embedContent' in m.supported_generation_methods:
                embed = m.name
            if not chat and 'generateContent' in m.supported_generation_methods:
                chat = m.name
        return embed, chat
    except Exception:
        return None, None

def save_chat_to_file():
    """Save current chat messages to a JSON file."""
    if st.session_state.messages:
        # Use first few words of the first message as filename base
        first_msg = st.session_state.messages[0]["content"][:15].strip().replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{HISTORY_DIR}/Chat_{first_msg}_{timestamp}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)

def load_chat_from_file(filename):
    """Load chat messages from a JSON file."""
    filepath = os.path.join(HISTORY_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        st.session_state.messages = json.load(f)

# ===================================================================
# 4. SESSION STATE INITIALISATION
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

    # ---------- CHAT HISTORY ----------
    st.subheader("📜 Saved Histories")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Start New Chat"):
            save_chat_to_file()
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Memory"):
            st.session_state.messages = []
            st.rerun()

    # List existing history files
    history_files = os.listdir(HISTORY_DIR)
    if history_files:
        for f in sorted(history_files, reverse=True):
            # Only show the first 25 characters of the filename
            short_name = f[:25] + ("..." if len(f) > 25 else "")
            if st.button(f"📁 {short_name}", key=f):
                load_chat_from_file(f)
                st.rerun()
    else:
        st.info("No saved chats yet.")

    st.divider()

    # ---------- API KEY & PDF UPLOAD ----------
    api_key = st.text_input("Gemini API Key", value=DEFAULT_API_KEY, type="password")
    uploaded_files = st.file_uploader(
        "Source Technical Documents (PDFs)",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("🚀 Initialize System"):
        if not api_key:
            st.error("Missing API Key! Please add it to Secrets or enter it above.")
        elif not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner(f"🔧 Processing {len(uploaded_files)} file(s)..."):
                all_chunks = []

                for uploaded_file in uploaded_files:
                    # Save temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Load and chunk
                    loader = PyMuPDFLoader(
                        temp_path,
                        extract_images=True,
                        images_parser=RapidOCRBlobParser()
                    )
                    data = loader.load()

                    # Add source file name to metadata
                    for doc in data:
                        doc.metadata["source_file"] = uploaded_file.name

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=150
                    )
                    chunks = text_splitter.split_documents(data)
                    all_chunks.extend(chunks)

                    # Clean up
                    os.remove(temp_path)

                # Find models
                embed_name, chat_name = find_models(api_key)
                if not embed_name or not chat_name:
                    st.error("Could not find a suitable embedding or chat model. Check your API key.")
                    st.stop()
                st.session_state.chat_model = chat_name

                # Create embeddings and vector store
                embeddings = GoogleGenerativeAIEmbeddings(
                    model=embed_name,
                    google_api_key=api_key
                )

                # Remove old database if exists
                db_path = "./chroma_db"
                if os.path.exists(db_path):
                    shutil.rmtree(db_path)

                st.session_state.vector_db = Chroma.from_documents(
                    documents=all_chunks,
                    embedding=embeddings,
                    persist_directory=db_path
                )
                st.success(f"✅ System Online with {len(uploaded_files)} document(s)!")

    if st.button("🗑️ Reset All"):
        # Delete all saved chats and clear state
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
st.subheader("🤖 AI Research Assistant")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Query the knowledge base..."):
    if not st.session_state.vector_db:
        st.error("System Offline: Please initialize in the sidebar.")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})
            source_docs = retriever.invoke(prompt)

            llm = ChatGoogleGenerativeAI(
                model=st.session_state.chat_model,
                google_api_key=api_key,
                temperature=0.1
            )

            template = """Answer based only on context:
{context}

Question: {question}"""
            chain = (
                {
                    "context": lambda _: "\n\n".join(d.page_content for d in source_docs),
                    "question": RunnablePassthrough()
                }
                | ChatPromptTemplate.from_template(template)
                | llm
                | StrOutputParser()
            )

            response = chain.invoke(prompt)
            st.markdown(response)

            # Show source material
            with st.expander("🔍 View Source Material"):
                for i, doc in enumerate(source_docs):
                    st.info(f"**Source {i+1}:** {doc.page_content[:300]}...")

            st.session_state.messages.append({"role": "assistant", "content": response})