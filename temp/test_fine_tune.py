import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, pipeline
from peft import LoraConfig, get_peft_model, PeftModel
import json
import torch
import datasets

model_name = "CodeLlama-13b-Instruct-hf"

model_id = f"meta-llama/{model_name}"
fine_tuned_model = f"fine_tuning/fine_tuned_model/{model_name}_QLoRA"

batch_size = 6
learning_rate = 1e-4
epochs = 20

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
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA to model
peft_model = get_peft_model(model, lora_config)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load and preprocess dataset
dataset_path = '../dataset/filled_json_dataset.json'
with open(dataset_path, 'r') as f:
    data = json.load(f)

# Format the data using a chat template (this step depends on your specific needs)
formatted_data = tokenizer.apply_chat_template(data, tokenize=False, add_generation_prompt=True)

# Create a Dataset object
dataset = datasets.Dataset.from_dict({"inputs": formatted_data})

tokenizer.pad_token = tokenizer.eos_token


# Tokenize the formatted input data and create shifted output labels
def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples["inputs"], truncation=True, max_length=5000, padding=True)
    # Self-supervised learning
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs


tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir=fine_tuned_model,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
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

del peft_model
torch.cuda.empty_cache()

# Load dataset
file_path = '../dataset/filtered_ielts_writing_dataset_v2.csv'
data = pd.read_csv(file_path)
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    quantization_config=None,  # Adjust this according to your quantization setup
    torch_dtype=torch.float16,
    device_map='auto'
)

# Load fine-tuned model
model = PeftModel.from_pretrained(model, fine_tuned_model)

# Set up pipeline
ppl = pipeline(task="text-generation",
               model=model,
               tokenizer=tokenizer,
               max_new_tokens=256,
               device_map="auto",
               torch_dtype=torch.bfloat16)

# Prepare to collect LLM responses
llm_assessments = []

index = 1

for _, row in data.iterrows():
    # Prepare instruction with the specific question and essay
    instruction = [
        {
            "role": "system",
            "content": "You are a professional examiner who assesses students essays based on the questions.\nThe question will be provided within <question> and </question> labels.\nThe essay content will be provided within <essay> and </essay> labels, make sure you identify the essay correctly.\nOutput follow a json format:\n{\n    \"Comment\": \"\",\n    \"Overall\": \n}"
        },
        {
            "role": "user",
            "content": f"<question>\n{row['Question']}\n</question>\n<essay>\n{row['Essay']}\n</essay>"
        }
    ]

    # Get LLM response
    response = ppl(instruction)

    # Assuming the model returns a list of responses, get the first response
    llm_response = response[0]['generated_text'][2]['content']

    print(f'{index}\n{llm_response}')
    index += 1

    # Parse the JSON response (you may need to handle this carefully if the response isn't perfect JSON)
    try:
        llm_response_json = json.loads(llm_response)
    except json.JSONDecodeError:
        llm_response_json = {"Comment": "",
                             "Overall": ""}

    # Append the LLM assessment along with the original human assessment
    llm_assessments.append({
        "Question": row['Question'],
        "Essay": row['Essay'],
        "Human_Comment": row['Examiner_Commen'],
        "LLM_Comment": llm_response_json.get("Comment", ""),
        "Human_Overall": row['Overall'],
        "LLM_Overall": llm_response_json.get("Overall", "")
    })

# Convert to DataFrame
llm_assessments_df = pd.DataFrame(llm_assessments)

# Save to CSV for further comparison
output_file_path = f'dataset/llm_vs_human_assessment_{learning_rate}_{epochs}_13b.csv'
llm_assessments_df.to_csv(output_file_path, index=False)

print(f"LLM assessments and human assessments saved to {output_file_path}")


