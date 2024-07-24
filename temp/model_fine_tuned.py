from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
import torch

model = "CodeLlama-7b-Instruct-hf"
model_id = f"meta-llama/{model}"
fine_tuned_model = f"fine_tuning/fine_tuned_model/{model}_QLoRA"

# Define the quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map='auto'
)

model.resize_token_embeddings(len(tokenizer))

# Load fine-tuned model
model = PeftModel.from_pretrained(model, fine_tuned_model)

# Set up pipeline
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, truncation=True, max_length=512)

# Example instruction
instruction = [
    {"role": "system", "content": "context: user: 'my name is dehao'"},
    {"role": "user", "content": "hello"},
]

# Generate text
result = pipe(instruction)
print(result)
