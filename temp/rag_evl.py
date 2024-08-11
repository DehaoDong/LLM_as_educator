import json

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from transformers import pipeline
from model import CodeLlama
import prompt_engineering as pe

LLMs = [
    "meta-llama/CodeLlama-7b-Instruct-hf",
    "meta-llama/CodeLlama-13b-Instruct-hf"
]

embedding_models = [
    "sentence-transformers/all-MiniLM-L6-v2", # 720m
    "sentence-transformers/all-roberta-large-v1",
    "sentence-transformers/bert-base-nli-mean-tokens"
]

# directory = "dataset/squad_v2/articles"
# loader = DirectoryLoader(directory)
# documents = loader.load()
#
# # Split documents
# text_splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=128, separator=" ")
# split_documents = text_splitter.split_documents(documents)


model_name = "sentence-transformers/bert-base-nli-mean-tokens"
model = "meta-llama/CodeLlama-13b-Instruct-hf"

encode_kwargs = {"normalize_embeddings": False}
model_kwargs = {"device": "cuda:0"}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("Embedding model loaded")

persist_directory = f"eval/knowledge_base/{model_name.replace('/', '-')}"
db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
retriever = db.as_retriever()

ppl = pipeline(task="text-generation",
               model=model,
               max_new_tokens=128,
               device_map="auto",
               torch_dtype=torch.bfloat16)
llm = CodeLlama(ppl=ppl)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": pe.RAG_PROMPT_TEMPLATE}
)

# Load the QA pairs from the JSON file
qa_pairs_path = "../dataset/squad_v2/qa_pairs.json"
with open(qa_pairs_path, 'r', encoding='utf-8') as f:
    qa_pairs = json.load(f)

qa_pairs = qa_pairs[:100]

# Prepare to collect the results
results = []

index = 1

# Query the LLM for each question
for pair in qa_pairs:
    question = pair['question']
    correct_answer = pair['answer']

    # Get the LLM's answer
    llm_answer = qa.invoke(question)['result']

    # Collect the results
    results.append({
        "question": question,
        "llm_answer": llm_answer,
        "correct_answer": correct_answer
    })

    print(f'question {index} done'
          f'\nquestion: {question}'
          f'\nllm_answer: {llm_answer}'
          f'\ncorrect_answer: {correct_answer}')

    index+=1


# make sure output directory exists
import os
if not os.path.exists("../eval/qa_results"):
    os.makedirs("../eval/qa_results")

# Save the results to a JSON file
output_file = f"eval/qa_results/{model_name.replace('/', '-')}-{model.replace('/', '-')}.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Results saved to: {output_file}")
