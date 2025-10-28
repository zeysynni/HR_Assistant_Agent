# 🧠 HR Assistant

An AI-powered assistant that helps HR professionals analyze job postings and match them with suitable candidates from a CV database.

---

## 🚀 Overview

**HR Assistant** is an intelligent agent built using **LangChain**, **OpenAI**, and **Gradio**.  
It can:
- Accept a **Job URL** via the Gradio chat interface.  
- **Scrape** and extract detailed job descriptions from the webpage.  
- Retrieve candidate information from a **vector-based CV database** built from PDF resumes.  
- Use **Retrieval-Augmented Generation (RAG)** to reason about candidate-job fit.  
- Provide natural language responses about the job or candidates.

---

## 🧩 Features

- 💬 **Interactive Gradio chat UI**  
  Easily chat with the assistant, ask about job descriptions, or request candidate matches.

- 🔍 **LinkedIn Job Scraping**  
  Automatically logs in and extracts structured information from job postings.

- 📄 **RAG-based Candidate Search**  
  Converts candidate CVs (PDF files) parallel into embeddings and stores them in a **Chroma vector database** for semantic retrieval.

- 🤖 **LLM Agent with Tools**  
  Integrates multiple tools (scraping, retrieval, reasoning) into a unified conversational agent.

---

## ⚙️ Tech Stack

- **Python 3.11+**
- **LangChain** (`langchain`, `langchain_openai`, `langchain_chroma`, `langchain_community`)
- **OpenAI GPT Models**
- **Gradio**
- **Selenium**
- **BeautifulSoup4**
- **ChromaDB**
- **dotenv**

---

## 📁 Project Structure 

- **app.py**: Main entry point with Gradio UI.
- **assistant.py**: Defines the core Assistant class and agent logic.
- **assistant_tools.py**: Tool definitions (e.g., scraping, retrieval).
- **rag_db.py**: Builds and manages the vector-based CV database.
- **requirements.txt**: Project dependencies.
- **.env**: Environment variables (API keys, credentials).
- **cv_base**: Folder containing candidate CVs in PDF format.

---

## 🔧 Setup & Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/HR_Assistant.git
   cd HR_Assistant
   
2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windowsf
   
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   
4. **Set up environment variables**


5. **Add candidates CVs in the cv_base folder**
   
## 🧠 How It Works

- **Job Scraping**: When a job URL is provided, the agent’s `get_jd` tool uses **Selenium + BeautifulSoup** to extract the job description.

- **CV Retrieval**: The assistant uses the `DB` class to read candidate PDFs, chunk the content, and store embeddings in a **Chroma vector store**.

- **Reasoning**: The LLM (**OpenAI GPT model**) uses **RAG** to find relevant candidates and respond contextually to questions.

- **Chat Memory**: The assistant keeps track of previous user inputs using `ConversationBufferMemory`.

## 📜 License

This project is released under the MIT License.

