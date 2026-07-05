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
    loader = DirectoryLoader( #Directory loader class from Langchain_community.document_loaders module. This class is used to load all the files from a directory. It takes three arguments: path, glob, and loader_cls.
        path = docs_path,
        glob = "**/*.pdf", #glob is a pattern matching syntax that allows us to specify which files we want to load from the directory. In this case, we are loading all PDF files
        loader_cls = PyPDFLoader  # we are using the PyPDFLoader class to load PDF files.
    )

    documents = loader.load() #Load the documents from the directory
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .pdf fies found in the directory {docs_path}. Please add your company documents.")
    
    for i, doc in enumerate(documents):
        print(f"\nDocument {i + 1}:")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:200]}...")  # Print the first 200 characters of the content
        print(f" Metadata: {doc.metadata}")

    return documents


def main():
    print("Main function")

    #1 Load the documents
    documents = load_documents(docs_path = "Doc")

    #2 Split the documents into chunks
    #3 Convert the chunks into vector embeddings 
    #4 Store the embeddings in a vector database
    #5 Create a retriever from the database
    #6 Create a question answering chain using the retriever and a language model
    #7 Use a chain to answer questions based on documents 
    



if __name__ == "__main__":
    main()    




