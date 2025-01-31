import os
import sys
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
    create_structured_chat_agent
)
from langchain_core.tools import Tool
from langchain.memory import ConversationBufferMemory

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from main import load_model

load_dotenv()
llm = load_model()


def get_current_time(*args, **kwargs):
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%I: %M %p")

def search_wikipedia(query):
    # searches wikipedia and returns the summary of the first result
    from wikipedia import summary

    try:
        return summary(query, sentences=2)
    except Exception as e:
        return f"""
        I couldn't find any information on that
        Error: {e}
        """

# List of tools available for the agent
tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="useful for when you need to know the current time"
    ),
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="useful for when you need to know information about a certain topic"
    ),
]
# Pull the prompt template from the hub
prompt = hub.pull("hwchase17/react")
chat_prompt = hub.pull("hwchase17/structured-chat-agent")

# Create the ReAct agent using create_react_function
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True
)

# store chat in memory
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True
)

chat_agent = create_structured_chat_agent(
    llm=llm, tools=tools, prompt=chat_prompt
)
chat_agent_executor = AgentExecutor.from_agent_and_tools(
    agent=chat_agent,
    tools=tools,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)
# run the agent with a test query
response = agent_executor.invoke({"input": "what time is it?"})
print(f"Response: {response}")