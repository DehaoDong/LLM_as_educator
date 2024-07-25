from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/CodeLlama-7b-Instruct-hf")

chat = [
   {"role": "system", "content": "system prompt"},
   {"role": "user", "content": "user prompt"},
   {"role": "assistant", "content": "assistant prompt"}
]

chat_template = tokenizer.apply_chat_template(chat, tokenize=False)

print(chat_template)