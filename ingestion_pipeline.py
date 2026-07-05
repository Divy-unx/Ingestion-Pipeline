import os 
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader # importing these two diff classes would helps us  to read text files, docx files 
from langchain_text_splitters import CharacterTextSplitter # To chunk the doc
from langchain_openai import OpenAIEmbeddings #TO conveert the chunks into vector embeddings 
from langchain_chroma import Chroma # we can host the dB locally 
from dotenv import load_dotenv 

load_dotenv()

def load_documents(docs_path = "Doc"):
    #Load all the files from the Doc Directory 
    print(f"Loading documents from {docs_path}...")

    #Check if Doc directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f" The directory {docs_path} does not exists. Please create it and add yoour company files.")
    
    #Load all (.pdf) files from the Doc directory
    loader = DirectoryLoader(
        path = docs_path,
        glob = "**/*.pdf",
        loader_cls = PyPDFLoader
    )

    # PyPDFLoader returns one Document per page — merge pages per file
    raw_documents = loader.load()
    
    if len(raw_documents) == 0:
        raise FileNotFoundError(f"No .pdf fies found in the directory {docs_path}. Please add your company documents.")
    
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
        doc = Document(page_content=data["content"].strip(), metadata=data["metadata"])
        documents.append(doc)
    
    for i, doc in enumerate(documents):
        print(f"\nDocument {i + 1}:")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:200]}...")
        print(f" Metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=200, chunk_overlap=0):
    #Split the documents into smaller chunks with overlap
    print("Splitting documents into chunks...")
    text_splitter = CharacterTextSplitter( # text splitting class to split the documents into smaller chunks with overlap
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks):
            print(f"\n----Chunk {i + 1} ----")
            print(f" SOurce: {chunk.metadata['source']}")
            print(f" Length: {len(chunk.page_content)} characters")
            print(f" Content: ")
            print(chunk.page_content)
            print("-" * 50)
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks.")    
        

    return chunks

    
    


def main():
    print("Main function")

    #1 Load the documents
    documents = load_documents(docs_path = "Doc")

    
    #2 Split the documents into chunks
    chunks = split_documents(documents)


    #3 Convert the chunks into vector embeddings 
    #4 Store the embeddings in a vector database
    #5 Create a retriever from the database
    #6 Create a question answering chain using the retriever and a language model
    #7 Use a chain to answer questions based on documents 
    



if __name__ == "__main__":
    main()
