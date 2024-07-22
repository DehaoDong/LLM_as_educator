# Use a pipeline as a high-level helper
import torch
from transformers import pipeline


messages = [
    {"role": "system", "content": "context: user: 'my name is dehao'"},
    {"role": "user", "content": "hello"},
]

pipe = pipeline("text-generation",
                model="meta-llama/CodeLlama-7b-Instruct-hf",
                max_new_tokens=512,
                device_map="auto",
                # batch_size=8,
                torch_dtype=torch.bfloat16,
                )

response = pipe(messages)


print(response)
