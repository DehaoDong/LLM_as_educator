from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import json
import torch
import datasets


def fine_tune(model_name):
    try:
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
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply LoRA to model
        peft_model = get_peft_model(model, lora_config)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

        # Load and preprocess dataset
        dataset_path = 'fine_tuning/datasets/fine_tune_dataset.json'
        with open(dataset_path, 'r') as f:
            data = json.load(f)

        # Create inputs and labels from the dataset
        inputs = []
        labels = []

        for chat in data:
            # Extract system and user messages for inputs
            input_messages = [msg for msg in chat if msg["role"] in ["system", "user"]]
            input_text = tokenizer.apply_chat_template(input_messages, tokenize=False, add_generation_prompt=False)
            inputs.append(input_text)

            # Extract assistant message for labels
            assistant_message = [msg["content"] for msg in chat if msg["role"] == "assistant"][0]
            labels.append(assistant_message)

        # print(f'inputs: {inputs}')
        # print(f'labels: {labels}')

        # Create a Dataset object
        dataset = datasets.Dataset.from_dict({"inputs": inputs, "labels": labels})

        # Show dataset details
        # print("Sample Data:", dataset[0])

        # Tokenize the formatted input data
        def tokenize_function(examples):
            tokenized_inputs = tokenizer(examples["inputs"], truncation=True, max_length=512, padding='max_length')
            tokenized_labels = tokenizer(examples["labels"], truncation=True, max_length=512, padding='max_length')

            tokenized_inputs["labels"] = tokenized_labels["input_ids"]
            return tokenized_inputs

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=fine_tuned_model,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=10,
            weight_decay=0.01,
            fp16=False,  # Disable fp16 to avoid gradient scaling issues
        )

        # Trainer
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )

        # Train model
        trainer.train()

        # Save the model
        peft_model.save_pretrained(fine_tuned_model, save_embedding_layers=True)

        return True, "Fine-tuning completed successfully."

    except Exception as e:
        return False, str(e)


# Example usage:
if __name__ == "__main__":
    model_name = "CodeLlama-7b-Instruct-hf"
    success, message = fine_tune(model_name)
    print(message)
