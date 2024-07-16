from langchain.prompts import PromptTemplate


QA_CHAIN_PROMPT = PromptTemplate.from_template("""
Some context that might be helpful will be provided in <context>.
If you find the context helpful, answer the question based on the context. If not, ignore the context.
If you don't know the answer, just say you don't know and don't try to make up an answer.
Keep your answer concise, with a maximum of 5 sentences.
context: {context}
question: {question}
""")


