#--Imports--#
from langchain.tools import tool
import datetime

#This is A File where we write the function for AI tools

@tool
def Time():
    """This tool returns the current time"""
    return datetime.datetime.now()