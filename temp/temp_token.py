import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from transformers import pipeline

from knowledge_base import embeddings

embedding_models = [
    "sentence-transformers/all-MiniLM-L6-v2", # 500
    "sentence-transformers/all-roberta-large-v1", #1758
    "sentence-transformers/bert-base-nli-mean-tokens" #862
]

directory = "dataset/squad_v2/articles"
loader = DirectoryLoader(directory)
documents = loader.load()

# Split documents
text_splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=128, separator=" ")
split_documents = text_splitter.split_documents(documents)

for embedding_model in embedding_models:
    model_name = embedding_model
    encode_kwargs = {"normalize_embeddings": False}
    model_kwargs = {"device": "cuda:0"}
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    print("Embedding model loaded")

    persist_directory = f"eval/knowledge_base/{model_name.replace('/', '-')}"
    db = Chroma.from_documents(split_documents, embedding_model, persist_directory=persist_directory)