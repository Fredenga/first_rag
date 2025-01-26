from langchain_community.vectorstores import Qdrant
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("./data/data.pdf")

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

texts = text_splitter.split_documents(documents)

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

x = Qdrant.from_documents(
    texts,
    embeddings,
    url = url,
    collection_name = collection_name
)

