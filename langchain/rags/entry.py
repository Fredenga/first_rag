import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# Updated import to Pydantic v2 directly
from pydantic import BaseModel  

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from main import load_model

load_dotenv()

# File path handling
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
model = load_model()

query = "Explain the concept of concurrency control"
client = QdrantClient(
        url=url,
        prefer_grpc=False
)

db = Qdrant(
        client=client,
        embeddings=embeddings,
        collection_name=collection_name
)

def feed_database():
    Qdrant.from_documents(
        docs,
        embeddings,
        url=url,
        collection_name=collection_name
    )
    print("Data vectorized successfully")

def query_database():
    docs = db.similarity_search_with_score(
        query=query,
        score_threshold=0.3
    )
    return docs

def query_model():
    relevant_docs = query_database()
    combined_input = (
        "Here are some documents that might help you answer the questions: "
        + query
        + "\nRelevant Documents:\n"
        + "\n\n".join([doc[0].page_content for doc in relevant_docs])
    )

    messages = [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    print("\n--- Generated Response ---")
    print(result)
    print("\nContent only")
    print(result.content)

def chat_with_model():
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    # Create a history-aware retriever
    # This uses the LLM to help reformulate the question based on chat history
    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )
    # Answer question prompt
    # This system prompt helps the AI understand that it should provide concise answers
    # based on the retrieved context and indicates what to do if the answer is unknown
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know. Use three sentences maximum and keep the answer "
        "concise."
        "\n\n"
        "{context}"
    )
    # Create a prompt template for answering questions
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    # Create a chain to combine documents for question answering
    # `create_stuff_documents_chain` feeds all retrieved context into the LLM
    # Model and Prompt
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

    # Create a retrieval chain that combines the history-aware retriever and question answering
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    print("Start chatting with the AI. Type exit to end the conversation")
    chat_history = []
    while True:
        word = input("You: ")
        if word.lower() == "exit":
            break
        # Process the user's query through the retrieval chain
        result = rag_chain.invoke({"input": word, "chat_history": chat_history})

        # Display the AI's response
        print(f"AI: {result['answer']}")

        # Update the chat history
        chat_history.append(HumanMessage(content=word))
        chat_history.append(SystemMessage(content=result['answer']))


        

if __name__ == "__main__":
    chat_with_model()
