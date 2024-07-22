from langchain.prompts import PromptTemplate


QA_PROMPT_TEMPLATE = PromptTemplate.from_template("""
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


