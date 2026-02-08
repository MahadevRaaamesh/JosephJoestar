#--Python Libs--#    
import os
from dotenv import load_dotenv
load_dotenv()

#--LangChain Libs--#
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser


#--Import the tools--#

from AITools import Time

#--AI Agent Creation--#

#-- Prompt --#
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant with access to tools."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=os.environ["GEMINI_API_KEY"])

# Define Memory

# Define the tools available to the agent
tools = [Time]

# Create the Agent
agent = create_agent(
    llm,
    tools,
    system_prompt="You are a helpful AI assistant with access to tools. You need not need to use your tools, just be a pa"
)

def AIChain(prompt):
    return agent.invoke({"messages": [{"role": "user", "content": prompt}]})

