import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

from config import MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP, DB_NAME, CV_ROOT, K_CLOSEST

class DB:
    
    #MODEL = "gpt-4o-mini"
    embeddings = OpenAIEmbeddings()
    openai = OpenAI()
    
    def __init__(self, temperature=0):
        self.temperature = temperature
        self.text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        self.cvs = glob.glob(CV_ROOT)
        self.cv_contents = [] 
        self.candidate_names = []

    def set_llm(self):
        return ChatOpenAI(temperature=self.temperature, model=MODEL_NAME)

    def set_memory(self):
        return ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    def read_cvs(self, max_workers=8):
        """Read and process CVs in parallel using threads."""
        def process_cv(cv_path):
            loader = PyPDFLoader(cv_path)
            pages = loader.load()
            applicant_name = self.retrieve_name(pages[0].page_content)
            for page in pages:
                page.metadata["applicant_name"] = applicant_name
            return applicant_name, pages

        self.cv_contents.clear()
        self.candidate_names.clear()
    
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_cv, cv): cv for cv in self.cvs}
            for future in as_completed(futures):
                name, pages = future.result()
                if name:
                    self.candidate_names.append(name)
                    self.cv_contents.extend(pages)
    
        print(f"Loaded {len(self.cv_contents)} pages from {len(self.candidate_names)} CVs.")

    def get_vectorstore(self):
        """Divide texts into chunks."""
        chunks = self.text_splitter.split_documents(self.cv_contents)
        
        if os.path.exists(DB_NAME):
            Chroma(persist_directory=DB_NAME, embedding_function=self.embeddings).delete_collection()
        
        vectorstore = Chroma.from_documents(documents=chunks, embedding=self.embeddings, persist_directory=DB_NAME)
        return vectorstore

    def set_retriever(self):
        return self.get_vectorstore().as_retriever(search_kwargs={"k": K_CLOSEST})

    def get_conver_chain(self):
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
    
        response = self.openai.chat.completions.create(model=MODEL_NAME, messages=messages)
    
        return response.choices[0].message.content