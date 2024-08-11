from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, pipeline
from peft import LoraConfig, get_peft_model, PeftModel
import json
import torch
import datasets

model_name = "CodeLlama-7b-Instruct-hf"

model_id = f"meta-llama/{model_name}"
fine_tuned_model = f"fine_tuning/fine_tuned_model/{model_name}_QLoRA"

# Define the quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map='auto',
    quantization_config=quantization_config
)

# LoRA configuration
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA to model
peft_model = get_peft_model(model, lora_config)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load and preprocess dataset
dataset_path = '../fine_tuning/datasets/fine_tune_dataset.json'
with open(dataset_path, 'r') as f:
    data = json.load(f)

# Format the data using a chat template (this step depends on your specific needs)
formatted_data = tokenizer.apply_chat_template(data, tokenize=False, add_generation_prompt=True)

# Create a Dataset object
dataset = datasets.Dataset.from_dict({"inputs": formatted_data})

tokenizer.pad_token = tokenizer.eos_token

# Tokenize the formatted input data and create shifted output labels
def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples["inputs"], truncation=True, max_length=10000, padding=True)

    # Shift the input_ids right to create the labels
    labels = tokenized_inputs["input_ids"].copy()
    labels_shifted_right = labels[:, 1:]  # Shifted right by one token
    labels_shifted_right = tokenizer.pad_token_id + labels_shifted_right.tolist()  # Ensure padding for alignment

    # Assign shifted labels back to the dataset
    tokenized_inputs["labels"] = labels_shifted_right
    tokenized_inputs["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in label] for label in labels_shifted_right]

    return tokenized_inputs

# Apply the tokenization and shifting function to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir=fine_tuned_model,
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    fp16=True,
)

# Trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# Save the model
peft_model.save_pretrained(fine_tuned_model, save_embedding_layers=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map='auto'
)

# Load fine-tuned model
model = PeftModel.from_pretrained(model, fine_tuned_model)

# Set up pipeline
ppl = pipeline(task="text-generation",
               model=model,
               tokenizer=tokenizer,
               max_new_tokens=1024,
               device_map="auto",
               torch_dtype=torch.bfloat16)

instruction = [
            {"role": "user", "content": "Who are you?"},
        ]

response = ppl(instruction)
print(response)