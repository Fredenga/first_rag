from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import fastembed
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
class ChunkVectorStore:
    def __init__(self):
        self.__url = "http://localhost:6333"
        self.__collection_name = "doctors_db"
        self.vector_store =  QdrantClient(
            url=self.__url,
            prefer_grpc=False
        )

    def split_into_chunks(self, file_path):
        # Load PDF document using PyPDFLoader
        doc = PyPDFLoader(file_path).load()
        
        # Initialize RecursiveCharacterTextSplitter to split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,   # Chunk size in characters
            chunk_overlap=20   # Overlap between chunks
        )

        # Split the document into chunks
        chunks = text_splitter.split_documents(doc)
        
        # Filter out unnecessary metadata
        chunks = filter_complex_metadata(chunks)

        return chunks

    def store_to_vector_database(self, chunks):
        # Store the chunks into the Chroma vector database using FastEmbed embeddings
        
        QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=fastembed.FastEmbedEmbeddings(),
            url=self.__url,
            collection_name=self.__collection_name
        )
        
        
        # Return the vector store object for further usage (e.g., querying)
        return self.vector_store
