# Use a pipeline as a high-level helper
import torch
from transformers import pipeline


messages = [
    {"role": "user", "content": "Who are you?"},
]

pipe = pipeline("text-generation",
                model="meta-llama/CodeLlama-7b-Instruct-hf",
                max_new_tokens=512,
                device_map="auto",
                # batch_size=8,
                torch_dtype=torch.bfloat16,
                )

response = pipe(messages)

response = response[0]['generated_text'][1]['content'].strip()

print(response)

# [
#     {'generated_text':
#         [
#             {'role': 'user', 'content': 'Who are you?'},
#             {'role': 'assistant', 'content': '  I am LLaMA, an AI assistant developed by Meta AI that can understand and respond to human input in a conversational manner. I am trained on a massive dataset of text from the internet and can generate human-like responses to a wide range of topics and questions. I can be used to create chatbots, virtual assistants, and other applications that require natural language understanding and generation capabilities.'}
#         ]
#     }
# ]
