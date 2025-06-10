import os
import traceback
from django.conf import settings
from PyPDF2 import PdfReader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FAISSLoader:
    db = None  # Store FAISS database

    @staticmethod
    def preload_faiss():
        try:
            print("Starting FAISS database preload...")

            # pdf_path = r'./../Knowledge_Base.pdf'

            pdf_path = os.path.join(settings.BASE_DIR, "Knowledge_Base.pdf")
            # Check if the PDF file exists
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            # Load the PDF
            pdf_reader = PdfReader(pdf_path)
            text = ""
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
            print("\n1. PDF extracted\n")

            # Split the text into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            print("\n2. Text split into chunks\n")

            # Initialize embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            # Create FAISS vector store
            vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
            print("\n3. Vector store created\n")

            # Store FAISS database
            FAISSLoader.db = vectorstore
            print("FAISS database successfully preloaded!")

        except Exception as e:
            print("Error during FAISS database preload:")
            print(traceback.format_exc())

    @staticmethod
    def get_faiss_db():
        if FAISSLoader.db is None:
            raise ValueError("FAISS database has not been preloaded!")
        return FAISSLoader.db
