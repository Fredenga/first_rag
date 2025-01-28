import os
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

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

def feed_database():
    Qdrant.from_documents(
        docs,
        embeddings,
        url = url,
        collection_name = collection_name
    )
    print("Data vectorized successfully")

def query_database():
    client = QdrantClient(
        url=url,
        prefer_grpc=False
    )

    db = Qdrant(
        client=client,
        embeddings=embeddings,
        collection_name=collection_name
    )

    query = "Explain the concept of concurrency control and provide an example"

    docs = db.similarity_search_with_score(query=query, k=5, score_threshold=0.5)
    
    for i in docs:
        doc, score = i
        print(f"score: {score},\n content: {doc.page_content},\n metadata: {doc.metadata}\n")

if __name__ == "__main__":
    feed_database()