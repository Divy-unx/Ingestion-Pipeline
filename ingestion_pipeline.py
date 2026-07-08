from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader  # importing these two diff classes would helps us  to read text files, docx files
from langchain_text_splitters import RecursiveCharacterTextSplitter  # To chunk the doc
from langchain_chroma import Chroma  # we can host the dB locally
from dotenv import load_dotenv



try:
    from langchain_huggingface import HuggingFaceEmbeddings
    EMBEDDING_MODEL_ARG = "model"
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    EMBEDDING_MODEL_ARG = "model_name"

PROJECT_ROOT = Path(__file__).resolve().parent
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE_DIR = PROJECT_ROOT / "models"

load_dotenv(PROJECT_ROOT / ".env")


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_documents(docs_path="Doc"):
    #Load all the files from the Doc Directory
    print(f"Loading documents from {docs_path}...")

    docs_path = project_path(docs_path)

    #Check if Doc directory exists
    if not docs_path.exists():
        raise FileNotFoundError(
            f"The directory {docs_path} does not exist. Please create it and add your company files."
        )

    #Load all (.pdf) files from the Doc directory
    loader = DirectoryLoader(
        path=str(docs_path),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    # PyPDFLoader returns one Document per page — merge pages per file
    raw_documents = loader.load()

    if len(raw_documents) == 0:
        raise FileNotFoundError(
            f"No .pdf files found in the directory {docs_path}. Please add your company documents."
        )

    # Group pages by source file and merge them into one Document per file
    from collections import OrderedDict
    merged = OrderedDict()

    for doc in raw_documents:
        source = doc.metadata["source"]
        if source not in merged:
            merged[source] = {"content": "", "metadata": doc.metadata.copy()}
        merged[source]["content"] += doc.page_content + "\n\n"

    documents = []
    for source, data in merged.items():
        from langchain_core.documents import Document
        doc = Document(
            page_content=data["content"].strip(),
            metadata=data["metadata"]
        )
        documents.append(doc)

    for i, doc in enumerate(documents):
        print(f"\nDocument {i + 1}:")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:200]}...")
        print(f" Metadata: {doc.metadata}")

    return documents


def split_documents(documents, chunk_size=500, chunk_overlap=150):
    #Split the documents into smaller chunks with overlap
    print("Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


def create_embeddings(chunks, persistence_dir="db/chroma_db"):
    #Convert the chunks into vector embeddings and store them in a vector database
    print("Creating embeddings and storing them in ChromaDB...")

    embedding_kwargs = {
        EMBEDDING_MODEL_ARG: EMBEDDING_MODEL,
        "cache_folder": str(MODEL_CACHE_DIR),
        "model_kwargs": {"device": "cpu", "local_files_only": True},
        "encode_kwargs": {"normalize_embeddings": True},
    }

    embeddings_model = HuggingFaceEmbeddings(**embedding_kwargs)
    print("Embedding model loaded successfully.")

    #Create a Chroma vector database from the chunks and embeddings
    print("Creating Vector Store...")
    # It takes all the chunks and converts into vector embeddings using the embedding model and stores in a Chroma Vector dB.

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=str(project_path(persistence_dir)),
        collection_metadata={"hnsw:space": "cosine"},
    )

    print("----Finished creating vector store----")
    print(f"Vector store created and saved to {project_path(persistence_dir)}.")

    if hasattr(vectorstore, "persist"):
        vectorstore.persist()

    return vectorstore


def main():
    print("Main function")

    #1 Load the documents
    documents = load_documents(docs_path="Doc")

    #2 Split the documents into chunks
    chunks = split_documents(documents)

    #3 Convert the chunks into vector embeddings
    vectorstore = create_embeddings(chunks)

    #4 Store the embeddings in a vector database
    #5 Create a retriever from the database
    #6 Create a question answering chain using the retriever and a language model
    #7 Use a chain to answer questions based on documents


if __name__ == "__main__":
    main()
