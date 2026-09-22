from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


def load_pdf(pdf_path):
    """
    Load a PDF and return its pages as LangChain Document objects.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents


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

def create_embeddings():
    """
    Create the embedding model used by Sensei.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    return embeddings


if __name__ == "__main__":

    # Step 1: Load PDF
    documents = load_pdf("sample_notes.pdf")

    print("Number of pages:", len(documents))

    # Step 2: Split into chunks
    chunks = split_documents(documents)

    print("Number of chunks:", len(chunks))

    # Step 3: Inspect the first few chunks
    for i, chunk in enumerate(chunks[:5]):
        print("\n" + "=" * 60)
        print(f"CHUNK {i + 1}")
        print("=" * 60)
        print(chunk.page_content)

    # Step 4: Create embedding model
    embeddings = create_embeddings()

    # Step 5: Convert the first chunk into a vector
    vector = embeddings.embed_query(chunks[0].page_content)

    print("\nEmbedding test")
    print("Vector dimensions:", len(vector))
    print("First 10 values:", vector[:10])