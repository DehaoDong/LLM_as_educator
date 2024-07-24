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

        # Load and process the JSON dataset
        with open("fine_tuning/datasets/fine_tune_dataset.json") as f:
            data = json.load(f)

        inputs = []
        outputs = []

        for conversation in data:
            prompt = "<s>"
            for message in conversation[:-1]:  # Exclude the last message which is the assistant's response
                role = message["role"]
                content = message["content"].strip()
                prompt += f"Source: {role}\n\n{content} <step> "

            # Add the destination for the model's response
            prompt += "Source: assistant\nDestination: user\n\n"

            # The last message is the assistant's response
            assistant_response = conversation[-1]["content"].strip()

            inputs.append(prompt.strip())
            outputs.append(assistant_response.strip())

        print(f'inputs: {inputs}\n')
        print(f'outputs: {outputs}\n')

        # Tokenize the dataset
        model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
        labels = tokenizer(outputs, truncation=True, padding="max_length", max_length=512)
        model_inputs["labels"] = labels["input_ids"]

        # print(f'model_inputs: {model_inputs}\n')

        # Convert to dataset format
        tokenized_dataset = datasets.Dataset.from_dict(model_inputs)

        # print(f'tokenized_dataset: {tokenized_dataset}\n')

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
