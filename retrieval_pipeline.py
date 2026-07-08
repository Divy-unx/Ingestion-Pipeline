import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "/tmp/ingestion_pipeline_models"))
groq_model = os.getenv("groq_model", "llama-3.3-70b-versatile")

load_dotenv(PROJECT_ROOT / ".env")

persistent_directory = str(PROJECT_ROOT / "db/chroma_db")
groq_api_key = os.getenv("groq_api_key")

if not groq_api_key:
    raise ValueError("groq_api_key is missing. Add it to your .env file.")

# Load embeddings and vector store
embedding_model = HuggingFaceEmbeddings(
    model=EMBEDDING_MODEL,
    cache_folder=str(MODEL_CACHE_DIR),
    model_kwargs={"device": "cpu", "local_files_only": True},
    encode_kwargs={"normalize_embeddings": True},
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# Search for relevant documents
query = "In what year did Tesla begin production of the Roadster?"

retriever = db.as_retriever(search_kwargs={"k": 5})

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.5
    }
)

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

context = "\n\n".join(doc.page_content for doc in relevant_docs)

llm = ChatGroq(
    model=groq_model,
    api_key=groq_api_key,
    temperature=0,
)

prompt = f"""Answer the question using only the context below.

Context:
{context}

Question:
{query}
"""

try:
    response = llm.invoke(prompt)
except Exception as exc:
    raise ValueError("Groq API request failed. Check that groq_api_key in .env is a valid Groq API key.") from exc

print("--- Answer ---")
print(response.content)


# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. ""
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"
