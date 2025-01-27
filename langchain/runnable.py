from main import load_model, define_template
from langchain.schema.runnable import RunnableLambda, RunnableSequence

model = load_model()
prompt_template = define_template()

# Inject provided values
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# RunnableLambda: A task that is a lambda function
# RunnableSequence: specifies the order that the Runnables should be executed

chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

response = chain.invoke({"topic": "construction", "content": "suspension bridges"})
print(response)