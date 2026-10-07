from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


# ============================================================
# Environment
# ============================================================

load_dotenv()

PERSIST_DIR = "chroma_db"


# ============================================================
# PDF Loading
# ============================================================

def load_pdf(pdf_path):
    """
    Load a PDF and return its pages as LangChain Document objects.
    """

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    return documents


# ============================================================
# Document Splitting
# ============================================================

def split_documents(documents):
    """
    Split PDF documents into smaller overlapping chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    return chunks


# ============================================================
# Embeddings
# ============================================================

def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={
            "device": "cuda"
        }
    )
    return embeddings


# ============================================================
# ChromaDB
# ============================================================

def build_vector_store(chunks, embeddings):
    """
    Create a persistent Chroma vector store from document chunks.
    """

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    return vector_store

def ingest_pdf(pdf_path, embeddings):
    """
    Load a PDF, split it into chunks, and store them
    persistently in ChromaDB with deterministic IDs.
    """

    import hashlib

    documents = load_pdf(pdf_path)
    chunks = split_documents(documents)

    vector_store = Chroma(
        collection_name="sensei_documents",
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    document_name = pdf_path

    document_hash = hashlib.md5(
        document_name.encode("utf-8")
    ).hexdigest()

    chunk_ids = []

    for index, chunk in enumerate(chunks):

        chunk.metadata["source_pdf"] = document_name

        chunk_id = f"{document_hash}_{index}"

        chunk_ids.append(chunk_id)

    vector_store.add_documents(
        documents=chunks,
        ids=chunk_ids
    )

    return vector_store

def load_existing_vector_store(embeddings):
    """
    Load the existing persistent ChromaDB vector store.
    """

    vector_store = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    return vector_store


# ============================================================
# Retrieval
# ============================================================

def retrieve_documents(vector_store, query, k=6):
    """
    Retrieve the most relevant chunks from ChromaDB.
    """

    results = vector_store.similarity_search(
        query,
        k=k
    )

    return results


# ============================================================
# Gemini Answer Generation
# ============================================================

def generate_answer(query, documents):
    """
    Generate an answer using only the retrieved document context.

    Sensei distinguishes between:
    1. Information actually present in the notes.
    2. Questions mentioned in a question bank without their solutions.
    3. Information completely absent from the retrieved context.
    """

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are Sensei, a grounded AI study assistant.

Your job is to answer the user's question using ONLY the
information contained in the provided notes.

IMPORTANT RULES:

1. Do NOT use outside knowledge.
2. Do NOT invent, complete, or assume an answer.
3. Carefully distinguish between:
   - information that is actually explained in the notes
   - a question that is merely listed in the notes
   - information that is completely absent from the notes
4. If the notes contain the answer or explanation, answer it clearly.
5. If the notes only contain a question about the topic but do not
   provide its answer or explanation, say:

   "The provided notes contain this question, but they do not
   provide its solution or explanation."

6. If the requested information is not present in the provided
   context at all, say:

   "I don't know based on the provided notes."

7. Never answer from your own general knowledge.
8. Preserve the terminology used in the notes.
9. Be concise but educational.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{query}

Now answer the user strictly according to the rules above.
"""

    model = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash"
    )

    response = model.invoke(prompt)

    if isinstance(response.content, list):
        return response.content[0]["text"]

    return response.content