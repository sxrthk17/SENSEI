
import hashlib
import os
import re

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "sensei_documents"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

DEFAULT_TOP_K = 6
NEIGHBOR_WINDOW = 1


# ============================================================
# ACTIVE DOCUMENT
# ============================================================

# These are set whenever a PDF is processed.
# They allow app.py to keep using:
#
#     retrieve_documents(vector_store, question, k=6)
#
# without needing to pass document_name/document_id.

ACTIVE_DOCUMENT_ID = None
ACTIVE_DOCUMENT_NAME = None


# ============================================================
# PDF LOADING
# ============================================================

def load_pdf(pdf_path):
    """
    Load a PDF and return LangChain documents.
    """

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    loader = PyPDFLoader(pdf_path)

    return loader.load()


# ============================================================
# DOCUMENT ID
# ============================================================

def generate_document_id(pdf_path):
    """
    Generate a stable ID from the actual PDF contents.

    Same PDF contents -> same ID
    Different PDF contents -> different ID
    """

    hasher = hashlib.sha256()

    with open(pdf_path, "rb") as file:

        while True:

            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


# ============================================================
# DOCUMENT NAME
# ============================================================

def get_document_name(pdf_path):
    """
    Return only the filename.
    """

    return os.path.basename(pdf_path)


# ============================================================
# CHUNKING
# ============================================================

def split_documents(
    documents,
    document_id=None,
    document_name=None,
):
    """
    Split PDF pages into chunks and attach metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):

        chunk.metadata["chunk_index"] = index

        if document_id is not None:

            chunk.metadata["document_id"] = (
                document_id
            )

        if document_name is not None:

            chunk.metadata["source_pdf"] = (
                document_name
            )

        page = chunk.metadata.get("page")

        if page is not None:

            chunk.metadata["page_number"] = (
                page + 1
            )

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

def create_embeddings():
    """
    Create MiniLM embeddings on CUDA.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cuda"
        },
    )


# ============================================================
# VECTOR STORE
# ============================================================

def get_vector_store(embeddings):
    """
    Open the persistent Sensei Chroma collection.
    """

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )


# ============================================================
# DUPLICATE DOCUMENT DETECTION
# ============================================================

def document_already_ingested(
    vector_store,
    document_id,
):
    """
    Check whether this exact PDF already exists.
    """

    try:

        collection = vector_store._collection

        result = collection.get(
            where={
                "document_id": document_id
            },
            include=[
                "metadatas"
            ],
        )

        return bool(
            result.get("ids")
        )

    except Exception:

        return False


# ============================================================
# PDF INGESTION
# ============================================================

def ingest_pdf(
    pdf_path,
    embeddings,
):
    """
    Ingest a PDF into Chroma.

    IMPORTANT:
    Returns ONLY the Chroma vector store so that it matches
    the current app.py contract.

    The active document ID/name are stored internally.
    """

    global ACTIVE_DOCUMENT_ID
    global ACTIVE_DOCUMENT_NAME


    # --------------------------------------------------------
    # Document identity
    # --------------------------------------------------------

    document_id = generate_document_id(
        pdf_path
    )

    document_name = get_document_name(
        pdf_path
    )


    # --------------------------------------------------------
    # Open vector store
    # --------------------------------------------------------

    vector_store = get_vector_store(
        embeddings
    )


    # --------------------------------------------------------
    # Mark this as the active document
    # --------------------------------------------------------

    ACTIVE_DOCUMENT_ID = document_id
    ACTIVE_DOCUMENT_NAME = document_name


    # --------------------------------------------------------
    # Duplicate protection
    # --------------------------------------------------------

    if document_already_ingested(
        vector_store,
        document_id,
    ):

        return vector_store


    # --------------------------------------------------------
    # Load PDF
    # --------------------------------------------------------

    documents = load_pdf(
        pdf_path
    )


    # --------------------------------------------------------
    # Split into chunks
    # --------------------------------------------------------

    chunks = split_documents(
        documents,
        document_id=document_id,
        document_name=document_name,
    )


    # --------------------------------------------------------
    # Unique chunk IDs
    # --------------------------------------------------------

    chunk_ids = []

    for index in range(len(chunks)):

        chunk_id = (
            f"{document_id}_{index}"
        )

        chunk_ids.append(
            chunk_id
        )


    # --------------------------------------------------------
    # Store in Chroma
    # --------------------------------------------------------

    vector_store.add_documents(
        documents=chunks,
        ids=chunk_ids,
    )


    return vector_store


# ============================================================
# LOAD EXISTING STORE
# ============================================================

def load_existing_vector_store(
    embeddings,
):
    """
    Load existing persistent Chroma database.
    """

    return get_vector_store(
        embeddings
    )


# ============================================================
# QUERY CLASSIFICATION
# ============================================================

DOCUMENT_LEVEL_PATTERNS = [

    r"\bhow many\b",
    r"\bcount\b",
    r"\btotal\b",
    r"\bnumber of\b",
    r"\bhow much\b",

    r"\bhow many pages\b",
    r"\bhow many questions\b",
    r"\bhow many mcq\b",
    r"\bhow many mcqs\b",

    r"\blist all\b",
    r"\blist the\b",
    r"\bshow all\b",
    r"\bwhat are all\b",

    r"\bwhich topics\b",
    r"\btopics covered\b",
    r"\bsections\b",
    r"\bchapters\b",

    r"\bsummarize the document\b",
    r"\bsummarise the document\b",

    r"\bsummarize this pdf\b",
    r"\bsummarise this pdf\b",

    r"\bentire document\b",
    r"\bwhole document\b",
]


def is_document_level_query(query):
    """
    Determine whether a query requires broad
    document-level evidence.
    """

    query = query.lower().strip()

    for pattern in DOCUMENT_LEVEL_PATTERNS:

        if re.search(
            pattern,
            query,
        ):

            return True

    return False


# ============================================================
# RETRIEVE ENTIRE ACTIVE DOCUMENT
# ============================================================

def retrieve_entire_document(
    vector_store,
    document_id=None,
    document_name=None,
):
    """
    Retrieve ALL chunks from ONLY the active document.
    """

    if not document_id and not document_name:

        return []


    collection = vector_store._collection


    # --------------------------------------------------------
    # Prefer document ID
    # --------------------------------------------------------

    try:

        if document_id:

            result = collection.get(
                where={
                    "document_id": document_id
                },
                include=[
                    "documents",
                    "metadatas",
                ],
            )

        else:

            result = collection.get(
                where={
                    "source_pdf": document_name
                },
                include=[
                    "documents",
                    "metadatas",
                ],
            )

    except Exception:

        return []


    documents = []

    raw_documents = result.get(
        "documents",
        [],
    )

    raw_metadatas = result.get(
        "metadatas",
        [],
    )


    for text, metadata in zip(
        raw_documents,
        raw_metadatas,
    ):

        documents.append(
            Document(
                page_content=text,
                metadata=metadata or {},
            )
        )


    # --------------------------------------------------------
    # Safety filter
    # --------------------------------------------------------

    safe_documents = []

    for document in documents:

        metadata = document.metadata

        if document_id:

            if metadata.get(
                "document_id"
            ) != document_id:

                continue

        elif document_name:

            if metadata.get(
                "source_pdf"
            ) != document_name:

                continue

        safe_documents.append(
            document
        )


    # --------------------------------------------------------
    # Restore chunk order
    # --------------------------------------------------------

    safe_documents.sort(
        key=lambda doc: doc.metadata.get(
            "chunk_index",
            0,
        )
    )


    return safe_documents


# ============================================================
# NEIGHBOR CHUNK EXPANSION
# ============================================================

def _expand_with_neighbors(
    vector_store,
    documents,
    window=NEIGHBOR_WINDOW,
):
    """
    Add neighboring chunks from the SAME document only.
    """

    if not documents:

        return []


    collection = vector_store._collection

    selected_ids = set()


    # --------------------------------------------------------
    # Find neighboring chunk IDs
    # --------------------------------------------------------

    for document in documents:

        metadata = document.metadata

        document_id = metadata.get(
            "document_id"
        )

        chunk_index = metadata.get(
            "chunk_index"
        )

        if (
            document_id is None
            or chunk_index is None
        ):

            continue


        for offset in range(
            -window,
            window + 1,
        ):

            target_index = (
                chunk_index + offset
            )

            if target_index < 0:

                continue


            target_id = (
                f"{document_id}_{target_index}"
            )

            selected_ids.add(
                target_id
            )


    if not selected_ids:

        return documents


    # --------------------------------------------------------
    # Fetch exact IDs
    # --------------------------------------------------------

    try:

        result = collection.get(
            ids=list(selected_ids),
            include=[
                "documents",
                "metadatas",
            ],
        )

    except Exception:

        return documents


    expanded = []

    for text, metadata in zip(
        result.get(
            "documents",
            [],
        ),
        result.get(
            "metadatas",
            [],
        ),
    ):

        expanded.append(
            Document(
                page_content=text,
                metadata=metadata or {},
            )
        )


    # --------------------------------------------------------
    # Restore order
    # --------------------------------------------------------

    expanded.sort(
        key=lambda doc: doc.metadata.get(
            "chunk_index",
            0,
        )
    )


    return expanded


# ============================================================
# SMART RETRIEVAL
# ============================================================

def retrieve_documents(
    vector_store,
    query,
    k=DEFAULT_TOP_K,
    document_name=None,
    document_id=None,
):
    """
    ACTIVE-DOCUMENT-ONLY retrieval.

    If document ID/name is not explicitly supplied,
    automatically use the document that was most recently
    processed by ingest_pdf().

    This makes the current app.py compatible while keeping
    document isolation enforced.
    """

    global ACTIVE_DOCUMENT_ID
    global ACTIVE_DOCUMENT_NAME


    # ========================================================
    # RESOLVE ACTIVE DOCUMENT
    # ========================================================

    if document_id is None:

        document_id = ACTIVE_DOCUMENT_ID


    if document_name is None:

        document_name = ACTIVE_DOCUMENT_NAME


    # ========================================================
    # HARD SAFETY CHECK
    # ========================================================

    if not document_id and not document_name:

        return []


    # ========================================================
    # DOCUMENT-LEVEL QUERY
    # ========================================================

    if is_document_level_query(query):

        return retrieve_entire_document(
            vector_store,
            document_id=document_id,
            document_name=document_name,
        )


    # ========================================================
    # BUILD CHROMA FILTER
    # ========================================================

    filter_metadata = None


    if document_id:

        filter_metadata = {
            "document_id": document_id
        }

    elif document_name:

        filter_metadata = {
            "source_pdf": document_name
        }


    if filter_metadata is None:

        return []


    # ========================================================
    # FILTERED SEMANTIC SEARCH
    # ========================================================

    try:

        documents = (
            vector_store.similarity_search(
                query,
                k=k,
                filter=filter_metadata,
            )
        )

    except Exception:

        return []


    # ========================================================
    # HARD SAFETY FILTER
    # ========================================================

    safe_documents = []

    for document in documents:

        metadata = document.metadata


        if document_id:

            if metadata.get(
                "document_id"
            ) != document_id:

                continue


        elif document_name:

            if metadata.get(
                "source_pdf"
            ) != document_name:

                continue


        safe_documents.append(
            document
        )


    # ========================================================
    # NEIGHBOR EXPANSION
    # ========================================================

    expanded_documents = (
        _expand_with_neighbors(
            vector_store,
            safe_documents,
        )
    )


    # ========================================================
    # FINAL HARD SAFETY FILTER
    # ========================================================

    final_documents = []

    seen = set()


    for document in expanded_documents:

        metadata = document.metadata


        # ----------------------------------------------------
        # Document isolation
        # ----------------------------------------------------

        if document_id:

            if metadata.get(
                "document_id"
            ) != document_id:

                continue

        elif document_name:

            if metadata.get(
                "source_pdf"
            ) != document_name:

                continue


        # ----------------------------------------------------
        # Duplicate protection
        # ----------------------------------------------------

        key = (
            metadata.get(
                "document_id"
            ),
            metadata.get(
                "chunk_index"
            ),
            document.page_content,
        )


        if key in seen:

            continue


        seen.add(key)

        final_documents.append(
            document
        )


    return final_documents


# ============================================================
# CONTEXT BUILDING
# ============================================================

def build_context(documents):
    """
    Convert retrieved documents into Gemini context.
    """

    if not documents:

        return ""


    context_parts = []


    for index, document in enumerate(
        documents,
        start=1,
    ):

        metadata = document.metadata

        source = metadata.get(
            "source_pdf",
            "unknown",
        )

        page = metadata.get(
            "page_number",
            "unknown",
        )

        chunk_index = metadata.get(
            "chunk_index",
            "unknown",
        )


        context_parts.append(
            f"""
--- SOURCE CHUNK {index} ---

PDF: {source}

Page: {page}

Chunk: {chunk_index}

{document.page_content}
"""
        )


    return "\n".join(
        context_parts
    )


# ============================================================
# GEMINI
# ============================================================

def create_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0,
    )


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(
    query,
    documents,
):
    """
    Generate an answer using ONLY retrieved evidence.
    """

    if not documents:

        return (
            "I don't know based on the provided notes."
        )


    context = build_context(
        documents
    )


    llm = create_llm()


    prompt = f"""
You are Sensei, a study assistant.

Your ONLY source of truth is the provided study material.

Do NOT use outside knowledge.
Do NOT guess.
Do NOT invent information.

IMPORTANT RULES:

1. Answer using ONLY the provided material.

2. If the answer is directly supported,
   answer clearly.

3. If the answer requires combining information
   from multiple parts of the provided material,
   combine that evidence carefully.

4. If the user asks for a count,
   determine the count from the provided material.

5. If the provided notes contain a question but
   do not contain its solution or explanation,
   say:

"The provided notes contain this question, but they do not provide its solution or explanation."

6. If the requested information cannot genuinely
   be determined from the provided material,
   say:

"I don't know based on the provided notes."

7. Never answer from general knowledge.

8. Never use information from another document.

9. Treat the supplied PDF material as the complete
   knowledge source for this answer.

USER QUESTION:

{query}

PROVIDED STUDY MATERIAL:

{context}

ANSWER:
"""


    response = llm.invoke(
        prompt
    )


    content = response.content


    # --------------------------------------------------------
    # Structured Gemini response
    # --------------------------------------------------------

    if isinstance(
        content,
        list,
    ):

        text_parts = []

        for item in content:

            if isinstance(
                item,
                dict,
            ):

                text = item.get(
                    "text"
                )

                if text:

                    text_parts.append(
                        text
                    )

            elif isinstance(
                item,
                str,
            ):

                text_parts.append(
                    item
                )


        return "\n".join(
            text_parts
        ).strip()


    if isinstance(
        content,
        dict,
    ):

        return str(
            content.get(
                "text",
                content,
            )
        ).strip()


    return str(
        content
    ).strip()


# ============================================================
# RETRIEVAL DIAGNOSTICS
# ============================================================

def get_retrieval_info(
    documents,
    query,
):
    """
    Return retrieval information for debugging/UI.
    """

    sources = sorted(
        {
            document.metadata.get(
                "source_pdf",
                "unknown",
            )
            for document in documents
        }
    )


    pages = sorted(
        {
            document.metadata.get(
                "page_number",
                "unknown",
            )
            for document in documents
        },
        key=str,
    )


    document_ids = sorted(
        {
            document.metadata.get(
                "document_id",
                "unknown",
            )
            for document in documents
        }
    )


    return {
        "query": query,

        "query_type": (
            "document-level"
            if is_document_level_query(query)
            else "local"
        ),

        "chunks_retrieved": len(
            documents
        ),

        "sources": sources,

        "document_ids": document_ids,

        "pages": pages,
    }
