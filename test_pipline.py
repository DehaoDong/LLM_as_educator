from fine_tune import fine_tune
from model import get_model_pipeline, CodeLlama
import prompt_engineering as pe


model = "CodeLlama-7b-Instruct-hf"

# success, message = fine_tune(model)
# print(success, message)

arguments = {
    "prompt": "Please introduce yourself.",
    "response": "I am Educator LLaMA, an AI assistant who specializes in computer science."
}

ad_prompt = pe.AD_PROMPT_TEMPLATE.format(**arguments)

ppl = get_model_pipeline(model)
llm = CodeLlama(ppl=ppl)

response = llm.invoke(ad_prompt)

print(response)
