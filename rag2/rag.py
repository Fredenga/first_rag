from chunk_vector_store import ChunkVectorStore as cvs
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

class Rag: 
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.cvs_obj = cvs()
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context
            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
                maximum and keep the answer concise. [/INST] </s>
            [INST] Question: {question}
            Context: {context}
            Answer: [/INST]
            """
        )
        self.model = ChatOllama(model="mistral")
        

    def feed(self, file_path: str):

        chunks = self.cvs_obj.split_into_chunks(file_path)

        self.vector_store = self.cvs_obj.store_to_vector_database(chunks)

        self.set_retriever()
        self.augment()


    def ask(self, prompt: str):
        pass

    def set_retriever(self):
        self.retriever = self.vector_store.as_retriever(
            search_type = "similarity_score_threshold",
            search_kwargs = {
                "k": 3,
                "score_threshold": 0.5

            }
        )

    def augment(self):
        pass