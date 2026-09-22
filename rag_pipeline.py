from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_path):
    """
    Load a PDF and return its pages as LangChain Document objects.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    return documents


if __name__ == "__main__":
    documents = load_pdf("sample_notes.pdf")

    print("Number of pages:", len(documents))
    print("\nFirst page:\n")
    print(documents[0].page_content[:1000])