# Load PDF
# Split into Chunks
# Create Embeddings
# Store in ChromaDB

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import     RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader("document loaders/deeplearning.pdf").load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunks = splitter.split_documents(data)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
  documents=chunks,
  embedding=embedding_model,
  persist_directory="./chroma_db" 
)