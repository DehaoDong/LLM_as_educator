from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

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
            target_modules=["q_proj", "v_proj"],  # Use the specific target modules you want to apply LoRA to
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply LoRA to model
        peft_model = get_peft_model(model, lora_config)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Add a padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            peft_model.resize_token_embeddings(len(tokenizer))

        # Save the tokenizer
        tokenizer.save_pretrained(fine_tuned_model)

        # Load dataset
        dataset = load_dataset("json", data_files="fine_tuning/datasets/fine_tune_dataset.json")

        # Tokenize the dataset
        def tokenize_function(examples):
            inputs = tokenizer(examples["user"], truncation=True, padding="max_length", max_length=512)
            outputs = tokenizer(examples["assistant"], truncation=True, padding="max_length", max_length=512)
            inputs["labels"] = outputs["input_ids"]
            return inputs

        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=fine_tuned_model,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            fp16=False,  # Disable fp16 to avoid gradient scaling issues
        )

        # Trainer
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
        )

        # Train model
        trainer.train()

        # Save the model
        peft_model.save_pretrained(fine_tuned_model, save_embedding_layers=True)

        return True, "Fine-tuning completed successfully."

    except Exception as e:
        return False, str(e)
