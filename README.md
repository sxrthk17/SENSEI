# SENSEI 🥋
### A Document-Grounded RAG Study Assistant

SENSEI is an AI-powered study assistant designed to help students learn from their own study materials. It combines document-grounded question answering, retrieval-augmented generation (RAG), adaptive quizzes, mathematical problem solving, and study productivity tools in one Streamlit application.

## ✨ Features

- **Notes Q&A:** Ask questions about uploaded PDF study materials and receive answers grounded in retrieved document passages.
- **Semantic Retrieval:** Find relevant passages using embedding-based similarity search.
- **Keyword-Based Reranking:** Reorder retrieved passages using keyword overlap alongside semantic ranking.
- **Source-Aware Answers:** Use document metadata, including source PDF, page, and chunk information, to support traceability.
- **Adaptive Quiz:** Practice concepts using the application's quiz functionality.
- **Math Solver:** Work through mathematical problems with the integrated math-solving module.
- **Study History:** Access persisted study-related records.
- **Pomodoro Focus:** Use a focus timer for structured study sessions.
- **To-do List:** Manage study tasks.
- **Persistent Storage:** Store supported application data locally using SQLite and document embeddings using ChromaDB.

## 🧠 How It Works

SENSEI's document-question-answering workflow follows these stages:

1. **Document ingestion:** Load PDF study materials.
2. **Text splitting:** Divide document content into overlapping chunks.
3. **Embedding generation:** Convert chunks into vector representations.
4. **Vector storage:** Persist embeddings and metadata in ChromaDB.
5. **Retrieval:** Search for semantically relevant chunks within the selected document scope.
6. **Reranking:** Combine semantic rank and keyword overlap to prioritize candidate passages.
7. **Answer generation:** Generate a response using the retrieved context.
8. **Source inspection:** Present retrieved passages and available source metadata for verification.

The current retrieval implementation reranks a bounded pool of semantic-search results. Independent keyword retrieval and broader retrieval-quality validation remain potential improvements.

## 🛠️ Technology Stack

- **Language:** Python
- **Application interface:** Streamlit
- **LLM integration:** LangChain Google GenAI
- **Embeddings:** Hugging Face Sentence Transformers
- **Vector database:** ChromaDB
- **PDF processing:** LangChain community document loaders and PyPDF
- **Local persistence:** SQLite
- **Mathematical operations:** SymPy

## 📁 Project Structure

```text
Sensei/
├── app.py
├── aesthetics.py
├── rag_pipeline.py
├── math_solver.py
├── memory_db.py
├── question_engine.py
├── requirements.txt
├── uploaded_notes/
│   └── library.json
├── chroma_db/
└── README.md
```

The database and storage directories may contain local application data. Preserve them when updating the application.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/sxrthk17/SENSEI.git
cd SENSEI
```

### 2. Create and activate a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If installation reports a missing package, install the corresponding dependency and update `requirements.txt` accordingly.

### 4. Configure environment variables

Create a `.env` file in the project root and configure the Google Gemini API key required by your current model setup:

```env
GOOGLE_API_KEY=your_google_api_key
```

If you enable optional web research, configure the corresponding Tavily API key as required by your implementation.

**Security:** Never commit API keys, credentials, or personal data to GitHub.

### 5. Run SENSEI

```bash
python -m streamlit run app.py
```

Streamlit will display the local URL for opening the application in your browser.

## 🧪 Current Verification Status

Initial manual checks of the Notes Q&A workflow have covered:

- Retrieving an exact formula from uploaded notes.
- Explaining a concept using retrieved study material.
- Recognizing when uploaded notes do not contain information needed to answer a question.

These checks are encouraging but do not constitute comprehensive testing. Multi-document isolation, retrieval ranking across a broader query set, performance on larger collections, and regression testing of all application features still require verification.

## 🔒 Data and Privacy

SENSEI uses local storage for supported application data and persistent document embeddings. External model or research services may receive submitted prompts or relevant context depending on the configured workflow. Review the implementation and service policies before using confidential or sensitive documents.

Do not delete the local database or vector-store directories as a troubleshooting shortcut; they may contain persisted application data.

## 🚧 Future Improvements

- Independent keyword retrieval combined with semantic retrieval.
- More systematic retrieval evaluation and regression tests.
- Improved retrieval performance for larger document collections.
- Further interface and user-experience refinement.
- Expanded documentation and release-readiness checks.

## 🎯 Project Goal

Build a reliable, document-grounded study assistant that helps learners understand their notes, practice concepts, solve mathematical problems, and manage their study workflow.

---

**SENSEI — Learn from your notes. Understand the concepts. Keep improving.**
