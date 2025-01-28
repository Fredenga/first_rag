import os
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "data", "Database_Systems.pdf")

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exist. Please check the path"
    )
# Load documents
loader = PyPDFLoader(file_path=file_path)
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents=documents)
# Load embeddings
 
import os

def load_embeddings():
    api_key = os.getenv("MISTRAL_API_KEY")

    embeddings = MistralAIEmbeddings(api_key=api_key)

    return embeddings

embeddings = load_embeddings()

url = "http://localhost:6333"
collection_name = "pdf_db"

x = Qdrant.from_documents(
    docs,
    embeddings,
    url = url,
    collection_name = collection_name
)