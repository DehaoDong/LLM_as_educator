import torch
from transformers import pipeline

instruction = [
    {
        "role": "user",
        "content": "What are some of the accepted general principles of European Union law?"
    }

]

ppl = pipeline(task="text-generation",
                            model="meta-llama/CodeLlama-13b-Instruct-hf",
                            max_new_tokens=1024,
                            device_map="auto",
                            torch_dtype=torch.bfloat16)

response = ppl(instruction)

print(response)
