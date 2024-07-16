from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM


class CustomLLM(LLM):
    n: int

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        print(f'Prompt: {prompt}')
        response = f'{prompt} {self.n}'
        print(f'Response: {response}')
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}


llm = CustomLLM(n=10)
response = llm.invoke("This is a foobar thing")
print(response)

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from model import CodeLlama, ModelHandler


def load_documents(directory="documents"):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    split_documents = text_splitter.split_documents(documents)

    return split_documents

def load_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    encode_kwargs = {"normalize_embeddings": False}
    model_kwargs = {"device": "cuda:0"}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def store_chroma(docs, embeddings, persist_directory="VectorStore"):
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    db.persist()
    return db


embeddings = load_embedding_model()

if not os.path.exists('VectorStore'):
    documents = load_documents()
    db = store_chroma(documents, embeddings)
else:
    db = Chroma(persist_directory='VectorStore', embedding_function=embeddings)

QA_CHAIN_PROMPT = PromptTemplate.from_template("""User:
Answer the question based on the following <context>.
If you don't know the answer, just say you don't know and don't try to make up an answer.
Keep your answer concise, with a maximum of 3 sentences.
Always end your answer by saying "Thank you for your question!"
{context}
question：{question}
Assistant:
""")

retriever = db.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    verbose=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

qa.run('what is the email address of Dehao Dong?')
