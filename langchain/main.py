from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI

import os

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

model = ChatMistralAI(api_key=api_key)

# Message for priming AI behavior, usually passed in as the first of a sequence of messages
# Gives broad context to the conversation (SystemMessage)
# Message from Human to AI model (HumanMessage)
# AI responding back to us (AIMessage)

# SystemMessage -> HumanMessage -> AIMessage

# messages = [
#     SystemMessage(content="Answer this history questions"),
#     HumanMessage(content="Why was Japan not split into two like Germany")
# ]

# result = model.invoke(messages)
# print(f"Answer from AI {result.content}")

chat_history = []
system_message = SystemMessage(content="You are a helpful AI assistant")
chat_history.append(system_message)

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=query)) # Add user message

    result = model.invoke(chat_history)
    response = result.content

    chat_history.append(AIMessage(content=response))
    print(f"AI {response}")
    
print("-----------Message History-----------")
print(chat_history)