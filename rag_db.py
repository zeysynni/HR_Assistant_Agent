import glob
from dotenv import load_dotenv
import os

from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv(override=True)
openai = OpenAI()

class DB:
    
    MODEL = "gpt-4o-mini"
    embeddings = OpenAIEmbeddings()
    
    def __init__(self, temperature=0, chunk_size=500, chunk_overlap=50, db_name="cv_db", cv_root="cv_base/*", k_closest=3):
        self.temperature = temperature
        self.text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.cvs = glob.glob(cv_root)
        self.db_name = db_name
        self.k_closest = k_closest
        self.cv_contents = [] 
        self.candidate_names = []

    def set_llm(self):
        return ChatOpenAI(temperature=self.temperature, model=self.MODEL)

    def set_memory(self):
        return ConversationBufferMemory(memory_key='rag_history', return_messages=True)

    def read_cvs(self):
        """Read in the content of CVs in the folder."""        
        for cv in self.cvs:
            loader = PyPDFLoader(cv)
            pages = loader.load() # 2 objects for 2 pages
            applicant_name = self.retrieve_name(pages[0].page_content)
            for page in pages:
                page.metadata["applicant_name"] = applicant_name
                self.cv_contents.append(page)
            self.candidate_names.append(applicant_name)

    def set_vectorstore(self):
        """Divide texts into chunks."""
        chunks = self.text_splitter.split_documents(self.cv_contents)
        
        if os.path.exists(self.db_name):
            Chroma(persist_directory=self.db_name, embedding_function=self.embeddings).delete_collection()
        
        vectorstore = Chroma.from_documents(documents=chunks, embedding=self.embeddings, persist_directory=self.db_name)
        return vectorstore

    def set_retriever(self):
        return self.set_vectorstore().as_retriever(search_kwargs={"k": self.k_closest})

    def set_conver_chain(self):
        self.read_cvs()
        llm = self.set_llm()
        retriever = self.set_retriever()
        memory = self.set_memory()
        conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
        return conversation_chain#, memory

    def user_prompt_for_name_retrievement(self, cv_content: str) -> str:
        """
        User prompt for retrieving name from a CV.
        Input:
            cv_content: str of 1 pdf content
        """
        user_prompt = "You are looking at a CV. \n"
        user_prompt += "\n The contents of this CV is as follows. \
        Please give only the name of the applicant. \n \
        Here is the CV: \n\n"
        user_prompt += cv_content
        return user_prompt

    def retrieve_name(self, cv_content: str) -> str:
        """
        Given the content of an applicant CV, return the name of the applicant.
        Args:
            cv_content: str of a CV.
        Returns:
            str: The name of the applicant.
        """
        system_prompt_name_retrievement = "You are an Assistant analyzes the contents of a CV \
        and provides the name of the CV holder. Ignoring text that might be part of a symbol. \
        Output only the name of the CV holder, nothing else."
    
        user_prompt = self.user_prompt_for_name_retrievement(cv_content)
    
        messages = [    
            {"role": "system", "content": system_prompt_name_retrievement},
            {"role": "user", "content": user_prompt}
        ]
    
        response = openai.chat.completions.create(model=self.MODEL, messages=messages)
    
        return response.choices[0].message.content