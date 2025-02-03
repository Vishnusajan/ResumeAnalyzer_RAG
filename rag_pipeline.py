import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import shutil
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class PDFQueryEngine:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        self.llm = ChatOpenAI(temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        self.vector_store = None
        self.persist_dir = "./chroma_db"
        self.job_description = None
        
        # Custom prompt template for resume analysis

        self.resume_prompt = PromptTemplate(
            template="""You are an expert HR professional analyzing resumes.

            Job Description:
            {job_description}

            Question: {query}

            """,input_variables=[ "query", "job_description"]
        )
        try:
            
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('averaged_perceptron_tagger')
        except:
            print("NLTK data already downloaded")
        
        self.stop_words = set(stopwords.words('english'))

    def set_job_description(self, description: str):
        """Set the job description for resume analysis"""
        self.job_description = self.preprocess_text(description)
        print("Job description set successfully")

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        text = text.lower()
        text = re.sub(r'\S+@\S+', ' EMAIL ', text)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', ' PHONE ', text)
        text = re.sub(r'http\S+|www.\S+', ' URL ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
        
    def load_pdf(self, pdf_path: str):
        if self.vector_store is not None:
            self.vector_store._client._collection.delete()
            self.vector_store = None
        
        try:
            if os.path.exists(self.persist_dir):
                shutil.rmtree(self.persist_dir)
        except Exception as e:
            print(f"Warning: Could not delete directory: {e}")
            
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        processed_documents = []
        for doc in documents:
            cleaned_text = self.preprocess_text(doc.page_content)
            doc.page_content = cleaned_text
            processed_documents.append(doc)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150
        )
        texts = text_splitter.split_documents(processed_documents)
        
        
        self.vector_store = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        print(f"Processed and loaded PDF with {len(texts)} text chunks")
    
    def ask_question(self, query: str):
        if not self.vector_store:
            raise ValueError("Please load a PDF first")
        
        if not self.job_description:
            raise ValueError("Please set a job description first using set_job_description()")
        
        formatted_prompt = self.resume_prompt.format(
            query=query,
            job_description=self.job_description
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )
        
        # Include job description in the query context
        return qa_chain.invoke(formatted_prompt)["result"]
__all__ = ['PDFQueryEngine']
