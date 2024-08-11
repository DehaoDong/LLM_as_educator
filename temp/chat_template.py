from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/CodeLlama-7b-Instruct-hf")

chat = [
  [
    {
      "role": "system",
      "content": "System prompt 1"
    },
    {
      "role": "user",
      "content": "User prompt 1"
    },
    {
      "role": "assistant",
      "content": "Assistant response 1"
    }
  ],
  [
    {
      "role": "system",
      "content": "System prompt 2"
    },
    {
      "role": "user",
      "content": "User prompt 2"
    },
    {
      "role": "assistant",
      "content": "Assistant response 2"
    }
  ]
]

chat_template_0 = tokenizer.apply_chat_template(chat[0], tokenize=False, add_generation_prompt=True)
chat_template_1 = tokenizer.apply_chat_template(chat[1], tokenize=False, add_generation_prompt=True)

print(f'Parsed data 0:\n{chat_template_0}\n'
      f'Parsed data 1:\n{chat_template_1}')