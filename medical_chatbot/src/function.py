from langchain.document_loaders import PyPDFDirectoryLoader ,PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO
from langchain_community.document_loaders import UnstructuredURLLoader


def load_data(data_path):
    loader = PyPDFDirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    data = loader.load()
    return data

def text_split(data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = splitter.split_documents(data)
    return text_chunks

def load_data_from_uploaded_pdf(file):
    loader = PyPDFDirectoryLoader(file)
    data = loader.load()
    return data

def load_data_from_url(url):
    loader = PyPDFDirectoryLoader(url)
    data = loader.load()
    return data

def load_data_from_url(url):
    url = [f"{url}"]
    loader = UnstructuredURLLoader(url)
    print("***************loader loaded***************")
    data = loader.load()
    return data

def text_split(data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = splitter.split_documents(data)
    return text_chunks

def download_huggingface_embedding():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

