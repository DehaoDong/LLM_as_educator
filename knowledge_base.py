import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from model import CodeLlama, ModelHandler


def load_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    encode_kwargs = {"normalize_embeddings": False}
    model_kwargs = {"device": "cuda:0"}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

embeddings = load_embedding_model()

def load_documents(directory="documents"):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=128, separator=r'\n| ', is_separator_regex=True)
    split_documents = text_splitter.split_documents(documents)

    # print(split_documents)

    return split_documents

def store_chroma(docs, persist_directory="VectorStore"):
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    db.persist()
    return db

def build_knowledge_base():
    print('Building knowledge base...')
    documents = load_documents()
    store_chroma(documents)

def get_knowledge_base_retriever():
    db = Chroma(persist_directory='VectorStore', embedding_function=embeddings)
    retriever = db.as_retriever()
    return retriever
