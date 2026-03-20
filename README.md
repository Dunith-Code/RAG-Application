# 📄 PDF Intelligence System
A Retrieval‑Augmented Generation (RAG) application that lets you upload one or more PDF documents and ask questions about their content. The system uses Google's Gemini models to embed your documents and generate accurate answers based solely on the provided context.

## 👇 Features
- Upload multiple PDFs – supports OCR (text extraction from images) via RapidOCRBlobParser.
- Ask questions in natural language – get answers grounded in the documents you've uploaded.
- Chat history – conversations are automatically saved and can be reloaded.
- Source citation – each answer includes expandable sections showing the exact chunks used.
- Dark‑themed interface – modern, easy‑on‑the‑eyes UI.
- Secure – API key stored in Streamlit Secrets, never hard‑coded.
- Deployable – ready to run on Streamlit Cloud or any Python environment.

## 🚀 How It Works

**1. Upload PDFs** – the app reads text from the files (including images) using PyMuPDFLoader and RapidOCRBlobParser.

**2. Chunking** – documents are split into overlapping chunks (default: 1000 characters, 150 overlap) to preserve context.

**3. Embedding** – each chunk is converted into a vector using a Gemini embedding model (auto‑detected from your API key).

**4. Vector store** – all vectors are stored in an in‑memory FAISS index (no database setup required).

**5. Retrieval** – when you ask a question, the app finds the 3 most similar chunks.

**6. Generation** – the chunks are injected into a prompt, and Gemini (chat model) generates an answer based only on that context.

**7. Chat persistence** – conversations are saved as JSON files in chat_sessions/; you can start new chats or load old ones.

## 🛠️ Installation
**1. Clone the repository**
```bash
git clone https://github.com/yourusername/pdf-intelligence-system.git
cd pdf-intelligence-system
```

**2. Set up a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up your API key**
* Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/api-keys).

* For local development, create a file .streamlit/secrets.toml with,
```toml
GOOGLE_API_KEY = "your-api-key-here"
```
* Never commit this file to version control – add .streamlit/ to your .gitignore.

**5. Run the app**
```bash
streamlit run app.py
```

## 📦 Requirements
All dependencies are listed in `requirements.txt`. Below is an overview,
|  Package | Purpose |
| :--- | :--- |
| `streamlit` | Web interface | 
| `google-generativeai` | Gemini API (models and embeddings) |
| `langchain` | RAG pipeline building blocks |
| `langchain-community` | Document loaders, FAISS integration |
| `langchain-text-splitters` | Chunking |
| `langchain-google-genai` | Gemini integration for LangChain |
| `pymupdf` | PDF text extraction (with OCR) |
| `rapidocr-onnxruntime` | OCR for images inside PDFs |
| `faiss-cpu` | Vector store (FAISS) |
| `pysqlite3-binary` | SQLite workaround for cloud deployment |

> **Note:** The `pysqlite3-binary` override is only needed if you deploy to Streamlit Cloud.

## 🖥️ Using the App
### Sidebar (Control Panel)
* **Saved Histories** – buttons to start a new chat, clear current chat, and load previous sessions.
* **Upload PDFs** – select one or more PDF files.
* **Initialize System** – processes the uploaded PDFs and builds the knowledge base. Wait for the success message.
* **Reset All** – deletes all chat history and the vector store.

### Main Chat Area
- Type your question in the text input.
- The assistant will respond with an answer and an expandable section showing the source chunks used.
- Your conversation is automatically saved when you start a new chat.


## 🌐 Deployment on Streamlit Cloud
1. Push your code to a GitHub repository.

2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and create a new app, linking to your repo.

3. In Advanced settings, add a secret,
    - Key: GOOGLE_API_KEY
    - Value: your Gemini API key

4. Click Deploy. The app will be live at `https://your-app-name.streamlit.app`.

> **Note:** The app uses `pysqlite3-binary` to work around SQLite version issues on the cloud. If you encounter any errors, check the logs via the “Manage app” menu.

## 🧪 Testing with Sample PDFs
You can test the system with the included sample files (if any) or with any PDF of your choice. For example, upload a technical paper or a product manual and ask questions like:

- What is the main argument of this paper?
- Summarise the section on mode‑locking.
- What are the differences between the two models?
  
## 🏗️ Project Structure
```text
pdf-intelligence-system/
├── app.py               # Main Streamlit application
├── requirements.txt     # Dependencies
├── chat_sessions/       # Saved conversations (created automatically)
├── .streamlit/
│   └── secrets.toml     # API key (ignored by git)
└── README.md            # This file
```

## 📝 Code Overview
The core logic is encapsulated in a few functions,

- `find_models()` – lists available Gemini embedding and chat models.
- `save_chat_to_file()` / `load_chat_from_file()` – persist conversations as JSON.
- Main `st.sidebar` block – handles PDF upload, chunking, embedding, and FAISS creation.
- Main chat block – retrieves, generates, and displays answers.

The vector store is built with FAISS, which keeps all vectors in memory, making the app fast and easy to deploy without external databases.

## 🔒 Security & Privacy
- API key – stored only in Streamlit Secrets, never exposed.
- User data – uploaded PDFs are temporarily saved and deleted after processing. Chat histories are saved locally on the server; you can delete them with the “Reset All” button.
- No tracking – the app does not collect any analytics or user data.

## 📜 License
This project is licensed under the MIT License – see the [LICENSE](https://github.com/Dunith-Code/RAG-Application/blob/main/LICENSE) file for details.

## 🤝 Acknowledgements

* **[Google Gemini API](https://ai.google.dev/)** — Provides the advanced language modeling and reasoning.
* **[LangChain](https://www.langchain.com/)** — The framework used to orchestrate the LLM and document chains.
* **[Streamlit](https://streamlit.io/)** — Powers the interactive web interface.
* **[FAISS](https://github.com/facebookresearch/faiss)** — Enables fast similarity search for the PDF vector embeddings.
* **[PyMuPDF](https://github.com/pymupdf/PyMuPDF)** — Handles high-performance PDF text and image extraction.

##
Happy Learning! 🚀
