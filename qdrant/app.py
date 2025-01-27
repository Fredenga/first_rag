from langchain_community.vectorstores import Qdrant
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

# Load the embedding model
model_name = "BAAI/bge-large-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

url = "http://localhost:6333"
collection_name = "gpt_db"

client = QdrantClient(
    url=url,
    prefer_grpc=False
)

print(client)

# For querying purposes
db = Qdrant(
    client=client,
    embeddings=embeddings,
    collection_name=collection_name
)

query = "How well does GPT4 perform when it comes to professional and academic exams?"

docs = db.similarity_search_with_score(query=query, k=5)

for i in docs:
    doc, score = i
    print(f"score: {score}, content: {doc.page_content}, metadata: {doc.metadata}")