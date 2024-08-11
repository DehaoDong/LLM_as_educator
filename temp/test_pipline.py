import torch
from transformers import pipeline

from fine_tune import fine_tune
from model import get_model_pipeline, CodeLlama
import prompt_engineering as pe


model = "CodeLlama-7b-Instruct-hf"

instruction = [
    {
        "role": "user",
        "content": "Who are you?"
    }

]

ppl = pipeline(task="text-generation",
                            model="meta-llama/CodeLlama-7b-Instruct-hf",
                            max_new_tokens=1024,
                            device_map="auto",
                            torch_dtype=torch.bfloat16)

response = ppl(instruction)

print(response)
