from chunk_vector_store import ChunkVectorStore as cvs
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

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


    def ask(self, query: str):
        self.augment()
        if not self.chain:
            raise Exception("Upload the document first to set the context of conversation")
        
        return self.chain.invoke(query)

    def set_retriever(self):
        self.retriever = self.vector_store.as_retriever(
            search_type = "similarity_score_threshold",
            search_kwargs = {
                "k": 3,
                "score_threshold": 0.5

            }
        ) 

    def augment(self):
        self.chain = (
            {
                "context": self.retriever,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.model
            | StrOutputParser()
        )