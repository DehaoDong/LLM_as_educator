from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Ensure PyTorch is set to use mixed precision
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True

# Load the model and tokenizer
model_name = "meta-llama/CodeLlama-7b-Instruct-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create the pipeline
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0  # Ensure this matches your GPU device
)

messages = [
    {"role": "user", "content": "Who are you?"},
]

response = pipe(messages,
                truncation=True,
                max_length=512)

# Print the result
print(response[0]['generated_text'][1]['content'])
