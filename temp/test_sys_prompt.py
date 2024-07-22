import torch
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
from model import CodeLlama
import knowledge_base as kb

llm = CodeLlama(pipeline = pipeline(task="text-generation",
                                model="meta-llama/CodeLlama-7b-Instruct-hf",
                                max_new_tokens=512,
                                device_map="auto",
                                # batch_size=8,
                                torch_dtype=torch.bfloat16,))




prompt_template = PromptTemplate.from_template("""
<SYS>
Some context that might be helpful will be provided below after 'context:'.
If you find the context helpful or relevant, answer the question based on the context. If not, ignore the context.
If you don't know the answer, just say you don't know and don't try to make up answers.
context: {context}
</SYS>
<USR>
{question}
</USR>
""")

retriever = kb.get_knowledge_base_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template}
)

print(f'QA: {qa}')

response = qa.invoke("What is VSDL?")

print(f'response: {response}')