import requests
from bs4 import BeautifulSoup
import PyPDF2
from langchain.document_loaders import PyPDFLoader,UnstructuredHTMLLoader
from langchain.schema import Document

class ContentLoader:
    @staticmethod
    def load_pdf(pdf_path):
        loader = PyPDFLoader(pdf_path)
        return loader.load()

    @staticmethod
    def load_website(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        return [Document(page_content=text, metadata={"source": url})]