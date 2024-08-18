from bs4 import BeautifulSoup
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline,BitsAndBytesConfig
from langchain_huggingface import HuggingFaceEmbeddings
from content_loader import ContentLoader


class RAGSystem:
    def __init__(self, model_name):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        self.embeddings = HuggingFaceEmbeddings()
        self.db = None
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.llm = HuggingFacePipeline(pipeline=pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.3,
            top_p=0.95
        ))

    def process_and_store(self):
        # Load content
        pdf_path='./Tymeline_White_Paper_Exploring_Team_Prod.pdf'
        website_url='https://www.tymeline.app/'
        pdf_docs = ContentLoader.load_pdf(pdf_path)
        web_docs = ContentLoader.load_website(website_url)
        all_docs = pdf_docs + web_docs

        # Split documents
        split_docs = self.text_splitter.split_documents(all_docs)

        # Create and store embeddings
        self.db = Chroma.from_documents(split_docs, self.embeddings)

    def answer_query(self, query):
        retriever = self.db.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(self.llm, retriever=retriever)
        return qa_chain.run(query)