import os
from dotenv import load_dotenv

from langchain.tools import tool

# Use selenium to scrap the LinkedIN page.
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import json, time
from dotenv import load_dotenv

from rag_db import DB

import warnings
warnings.filterwarnings("ignore")
import requests
from bs4 import BeautifulSoup

load_dotenv(override=True)
LinkedIN_Username = os.getenv('LinkedIN_Username')
LinkedIN_Password = os.getenv('LinkedIN_Password')

rag = DB()
conversation_chain = rag.set_conver_chain() 
candidate_names = rag.candidate_names

@tool
def get_jd(url: str) -> str:
    """
    Input:
        url: str of job content.
    Output:
        str of scrapped job description.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    return(soup.text)

@tool
def get_linkedIn_jd(url: str) -> str:
    """
    Input:
        url: str of job content on LinkedIn.
    Output:
        str of scrapped job description.
    """
    
    driver = webdriver.Chrome()
    driver.get("https://www.linkedin.com/login")
    
    driver.find_element(By.ID, "username").send_keys(LinkedIN_Username)
    driver.find_element(By.ID, "password").send_keys(LinkedIN_Password)
    driver.find_element(By.XPATH, "//button[@type='submit']").click()
    driver.get(url)
    time.sleep(2)
    
    try:
        mehr_anzeigen = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//button[.//span[text()='Mehr anzeigen']]"))
        )
    
        driver.execute_script("arguments[0].click();", mehr_anzeigen)
    
    
        WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//button[.//span[text()='Mehr anzeigen']]"))
        )
        
        mehr_anzeigen.click()
        print("Clicked 'Mehr anzeigen'")
    except Exception:
        print("Mehr anzeigen not found")

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    jd = soup.body.get_text(separator="/n", strip=True).split("}")[-1]
    parts = jd.split("Details zum Jobangebot")
    jd = parts[-1]
    jd = jd.split("Mehr anzeigen")
    driver.quit()
    return jd[0]

@tool
def rag_tool_fn(query: str) -> str:
    """Use this to retrieve candidates information from internal documents."""
    return conversation_chain.invoke({"question": query})["answer"]

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

@tool
def candidate_list_fn(query: str) -> list:
    """Returns a list of all available candidate names. Refer to it if you are asked about candidates, different candidates could have the same name. Check the details of the candidates if needed."""
    return candidate_names

def tools():
    return [get_jd, rag_tool_fn, fallback, candidate_list_fn]


