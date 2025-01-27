from main import load_model, define_template
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain.schema.output_parser import StrOutputParser

model = load_model()
prompt_template = define_template()

def use_runnables():
    # Inject provided values
    format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
    invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
    parse_output = RunnableLambda(lambda x: x.content)

    # RunnableLambda: A task that needs to be performed in the pipeline. It is formatted as a lambda function
    # RunnableSequence: specifies the order that the Runnables should be executed

    chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

    response = chain.invoke({"topic": "construction", "content": "suspension bridges"})
    print(response)

    # Shortcut: chain = prompt_template | model | StrOutputParser()

def use_chaining():
    uppercase_output = RunnableLambda(lambda x: x.upper())
    count_words = RunnableLambda(lambda x: f"{len(x.split())}\n{x}")

    chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

    response = chain.invoke({"topic": "companies", "content": "Amazon"})
    print(response)

if __name__ == "__main__":
    use_chaining()