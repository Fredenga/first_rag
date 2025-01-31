import os
import sys
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent
)
from langchain_core.tools import Tool
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from main import load_model

load_dotenv()
llm = load_model()


def get_current_time(*args, **kwargs):
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%I: %M %p")

# List of tools available for the agent
tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="useful for when you need to know the current time"
    )
]
# Pull the prompt template from the hub
prompt = hub.pull("hwchase17/react")

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
# run the agent with a test query
response = agent_executor.invoke({"input": "what time is it?"})
print(f"Response: {response}")