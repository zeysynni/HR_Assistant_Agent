from langchain.tools import tool

from bs4 import BeautifulSoup
import requests
import json, time
from dotenv import load_dotenv

from config import USER_AGENT

@tool
def get_job_description(url: str) -> str:
    """
    Use this to scrape a website, unless you see that website is a LinkedIn page.
    Input:
        url: str of job content.
    Output:
        str of scrapped job description.
    """
    headers = {"User-Agent": USER_AGENT}
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    return(soup.text)
    
def make_rag_tool(conversation_chain):
    @tool
    def rag_tool_fn(query: str) -> str:
        """Use this to retrieve candidates information from internal documents."""
        return conversation_chain.invoke({"question": query})["answer"]
    return rag_tool_fn

@tool
def fallback(query: str) -> str:
    """
    Fallback tool for handling queries that do not require any specific tool action.
    
    Args:
        query (str): The user input that does not match any other tool's purpose.
    
    Returns:
        str: A simple response echoing or acknowledging the input.
    
    Purpose:
        Ensures the agent always has a valid tool to call, preventing 
        parsing errors when the user input is casual or unrelated to
        defined tools.
    """
    return f"You said: {query}, but I don’t need to call a tool for this."

def make_candidate_list_tool(candidate_names):
    @tool
    def candidate_list_fn(query: str) -> list:
        """Returns a list of all available candidate names. Refer to it if you are asked about candidates, different candidates could have the
        same name. Check the details of the candidates if needed."""
        return candidate_names
    return candidate_list_fn

def init_tools(db):
    """
    Initialize all tools with dependencies from the given DB.
    
    Args:
        db (DB): An instance of your RAG database class.

    Returns:
        list: List of tool objects ready for the agent.
    """
    conversation_chain = db.get_conver_chain()
    candidate_names = db.candidate_names

    return [
        get_job_description,
        make_rag_tool(conversation_chain),
        make_candidate_list_tool(candidate_names),
        fallback,
    ]


