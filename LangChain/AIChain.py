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

from AITools import get_current_time, write_file, create_directory, read_file, list_directory_contents

#--AI Agent Creation--#

#-- Prompt --#

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=os.environ["GEMINI_API_KEY"])

# Define Memory

# Define the tools available to the agent
tools = [get_current_time, write_file, create_directory, read_file, list_directory_contents]

# Create the Agent
agent = create_agent(
    llm,
    tools,
    system_prompt="""You are a helpful AI assistant with access to tools. Your primary goal is to assist users in creating and managing simple websites.
    When asked to create a website, you should:
    1. Ask clarifying questions about the website's purpose, content, and desired design.
    2. Plan the file structure (e.g., 'website/' directory, 'index.html', 'style.css', 'script.js').
    3. Use the 'create_directory' tool to set up the necessary folders.
    4. Generate the HTML, CSS, and JavaScript content based on the user's request.
    5. Use the 'write_file' tool to save these contents into the appropriate files within the created directory.
    6. Inform the user about the created files and their location.
    7. If a deployment tool were available, you would then use it to deploy the website.
    You may explain your reasoning when the user asks.
    You should act like a teacher when asked to plan or explain.
    You should only use tools when necessary."""
)

def AIChain(prompt):
    return agent.invoke({"messages": [{"role": "user", "content": prompt}]})
