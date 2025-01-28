from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from main import load_model, define_template
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain.schema.output_parser import StrOutputParser

model = load_model()
prompt_template = define_template()

def analyze_pros(features):
    pros_template = ChatPromptTemplate([
        ("system", "You are an expert product reviewer"),
        ("human", "Given these features {features}, list the pros of these features")
    ])

    return pros_template.format_prompt(features=features)

def analyze_cons(features):
    cons_template = ChatPromptTemplate([
        ("system", "You are an expert product reviewer"),
        ("human", "Given these features {features}, list the cons of these features")
    ])

    return cons_template.format_prompt(features=features)

def combine_pros_and_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"

def parallel_chaining():
    pros_branch = (
        RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
    )

    cons_branch = (
        RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
    )
    
    chain = (
        prompt_template 
        | model 
        | StrOutputParser() 
        | RunnableParallel(branches={"pros": pros_branch, "cons": cons_branch})
        | RunnableLambda(lambda x: combine_pros_and_cons(x["branches"]["pros"], x["branches"]["cons"]))
    )
    # Parallel returns a branches dictionary, the last Lambda takes the values of this branches dict and combines them together in one output

    results = chain.invoke({"topic": "product review", "content": "MacBook Pro"})
    print(results)
  
if __name__ == "__main__":
    parallel_chaining()
