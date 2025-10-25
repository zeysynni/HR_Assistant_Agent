from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.messages import SystemMessage
from langchain_classic.agents import initialize_agent, AgentType
from langchain_core.prompts import MessagesPlaceholder
from langchain_classic.agents import create_openai_functions_agent, AgentExecutor
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import SystemMessagePromptTemplate

from rag_db import DB
from assistant_tools import init_tools
from config import MODEL_NAME, TEMPERATURE

rag = DB()

class Assistant:
    
    #MODEL = "gpt-4o-mini"
    
    def __init__(self, cv_root="cv_base/*", db_name="cv_db", chunk_size=500, chunk_overlap=50, temperature=0, k_closest=3):
        self.db_name = db_name
        #self.temperature = temperature
        self.llm = ChatOpenAI(temperature=TEMPERATURE, model=MODEL_NAME)
        
    def system_prompt_choose_candidate(self) -> str:
        """Create a system prompt for deciding wheather there are appropritate candidate for a given job.""" 
        system_prompt = "You are an expert in searching for ideal candidate for a job advertisement. \n"
        system_prompt += "If you give suggestions for suitable candidates, always make sure the following: \n "
        system_prompt += "1. Give the full name of appropriate candidates. \n"
        system_prompt += "2. Also give brief reasoning for you decision. \n" 
        system_prompt += "\n Always answer in English. \n\n"
        #system_prompt += f"You have CVs of these candidates: {candidate_names}.\n"
        system_prompt += "Only use a tool if you need to look up information. If the input is a greeting or casual chat, respond directly."
        return system_prompt


    def build_agent(self):
        llm = self.llm
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        memory.chat_memory.messages.insert(0, SystemMessage(content=self.system_prompt_choose_candidate()))
    
        # Build a custom prompt template that includes chat history
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt_choose_candidate()),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
    
        agent = create_openai_functions_agent(llm=llm, tools=init_tools(rag), prompt=prompt)
    
        agent_executor = AgentExecutor(
            agent=agent,
            tools=init_tools(rag),
            memory=memory,
            handle_parsing_errors=True,
            verbose=True,
        )
    
        return agent_executor
        
        
        
    