#--Imports--#
from langchain.tools import tool
import os
import datetime

#This is A File where we write the function for AI tools

@tool
def get_current_time() -> str:
    """This tool returns the current time in ISO 8601 format."""
    return datetime.datetime.now().isoformat()

@tool
def write_file(file_path: str, content: str) -> str:
    """
    Writes the given content to a specified file path.
    Example: write_file("index.html", "<h1>Hello World!</h1>")
    """
    try:
        if os.path.dirname(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)
        return f"Successfully wrote content to {file_path}"
    except Exception as e:
        return f"Error writing to file {file_path}: {e}"

@tool
def create_directory(directory_path: str) -> str:
    """Creates a new directory at the specified path."""
    os.makedirs(directory_path, exist_ok=True)
    return f"Directory '{directory_path}' created or already exists."

@tool
def read_file(file_path: str) -> str:
    """
    Reads the content of a specified file path and returns it as a string.
    Example: read_file("index.html")
    """
    try:
        with open(file_path, "r") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
    except Exception as e:
        return f"Error reading file {file_path}: {e}"

@tool
def list_directory_contents(directory_path: str = ".") -> str:
    """Lists the contents (files and subdirectories) of a specified directory path. Defaults to the current directory if no path is provided."""
    try:
        return "\n".join(os.listdir(directory_path))
    except FileNotFoundError:
        return f"Error: Directory not found at {directory_path}"
    except Exception as e:
        return f"Error listing directory contents for {directory_path}: {e}"