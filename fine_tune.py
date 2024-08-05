from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import json
import torch
import datasets


def fine_tune(model_name, learning_rate=1e-4, num_train_epochs=30):
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
        dataset_path = 'fine_tuning/datasets/fine_tune_dataset.json'
        with open(dataset_path, 'r') as f:
            data = json.load(f)

        # Format data
        # formatted_data = []
        # for chat in data:
        #     # print(f'chat: {chat}')
        #     formatted_chat = ""
        #     for msg in chat:
        #         # print(f'msg: {msg}')
        #         formatted_chat += f"<{msg['role']}>\n{msg['content']}\n</{msg['role']}>\n"
        #     formatted_data.append(formatted_chat.strip())

        formatted_data = tokenizer.apply_chat_template(data, tokenize=False, add_generation_prompt=True)

        # print(formatted_data)
        # print(f'length of formatted data: {len(formatted_data)}')

        # Create a Dataset object
        dataset = datasets.Dataset.from_dict({"inputs": formatted_data})

        tokenizer.pad_token = tokenizer.eos_token

        print(f'tokenizer.model_max_length: {tokenizer.model_max_length}')

        # Tokenize the formatted input data
        def tokenize_function(examples):
            tokenized_inputs = tokenizer(examples["inputs"], truncation=True, max_length=100000, padding=True)
            # Self-supervised learning
            tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
            return tokenized_inputs


        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=fine_tuned_model,
            learning_rate=learning_rate,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=num_train_epochs,
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

        return True, "Fine-tuning completed successfully."

    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    model_name = "CodeLlama-7b-Instruct-hf"
    success, message = fine_tune(model_name)
    print(message)
