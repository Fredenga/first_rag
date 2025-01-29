import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.messages import HumanMessage, SystemMessage
from qdrant_client import QdrantClient

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from main import load_model

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
    model_name = "BAAI/bge-large-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

embeddings = load_embeddings()

url = "http://localhost:6333"
collection_name = "pdf_db"
query = "Explain the concept of concurrency control"

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

    docs1 = db.similarity_search_with_score(query=query, k=5, score_threshold=0.5)

    docs2 = db.max_marginal_relevance_search(
        query, embeddings, k=3, fetch_k=20, lambda_mult=0.5
    )

    docs3 = db.similarity_search(
        query, embeddings, k=4
    )
    
    for i in docs1:
        doc, score = i
        print(f"score: {score},\n content: {doc.page_content},\n metadata: {doc.metadata}\n")

    return docs1

def query_model():
    relevant_docs = query_database()
    combined_input = (
        "Here are some documents that might help you answer the questions: "
        + query
        + "\nRelevant Documents:\n"
        + "\n\n".join([doc.page_content for doc in relevant_docs])
    )

    model = load_model()
    messages = [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    print("\n--- Generated Response ---")
    print(result)
    print("\nContent only")
    print(result.content)


if __name__ == "__main__":
    feed_database()