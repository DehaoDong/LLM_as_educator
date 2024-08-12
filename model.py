import json
import os
from datetime import datetime
from typing import Any, List, Mapping, Optional
from peft import PeftModel
from transformers import pipeline, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import torch
from langchain.llms.base import LLM
from concurrent.futures import ThreadPoolExecutor

HISTORY_FILE = 'history/history.json'
HISTORY_LIMIT = 20

MAX_NEW_TOKEN = 500

# For saving history in a separate thread
executor = ThreadPoolExecutor(max_workers=1)


def get_model_pipeline(model, is_finetuned=False):
    model_id = f"meta-llama/{model}"
    fine_tuned_model = f"fine_tuning/fine_tuned_model/{model}_QLoRA"

    if not is_finetuned or not os.path.exists(fine_tuned_model):
        print(f"Loading base {model}...")
        ppl = pipeline(task="text-generation",
                       model=model_id,
                       max_new_tokens=MAX_NEW_TOKEN,
                       device_map="auto",
                       torch_dtype=torch.bfloat16)
        return ppl
    else:
        print(f"Loading fine-tuned {model}...")
        # Define the quantization configuration
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )

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
                       max_new_tokens=MAX_NEW_TOKEN,
                       device_map="auto",
                       torch_dtype=torch.bfloat16)

        return ppl


class CodeLlama(LLM):
    ppl: Any

    @property
    def _llm_type(self) -> str:
        return str(self.ppl)

    def __extract_answer(self, result):
        generated_text = result[0]['generated_text']

        for message in generated_text:
            if message['role'] == 'assistant':
                return message['content'].strip()

        return None

    def __save_history(self, result):
        def save_to_file(result):
            # Load existing history if it exists
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, 'r') as f:
                    history = json.load(f)
            else:
                history = []

            # Append new result with current time
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'generated_text': result[0]['generated_text']
            }
            history.append(history_entry)

            # Limit history to the most recent 20 entries
            if len(history) > HISTORY_LIMIT:
                history = history[-HISTORY_LIMIT:]

            # Save updated history back to the file
            with open(HISTORY_FILE, 'w') as f:
                json.dump(history, f, indent=2)

            print(f'current history: {len(history)}')

        # Run the save_to_file function in a separate thread
        executor.submit(save_to_file, result)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        sys_prompt_start = prompt.find("<<SYS>>") + len("<<SYS>>")
        sys_prompt_end = prompt.find("<</SYS>>")
        usr_prompt_start = prompt.find("<<USR>>") + len("<<USR>>")
        usr_prompt_end = prompt.find("<</USR>>")

        system_prompt = prompt[sys_prompt_start:sys_prompt_end].strip()
        user_prompt = prompt[usr_prompt_start:usr_prompt_end].strip()

        instruction = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        result = self.ppl(instruction)
        print(json.dumps(result, indent=2))

        self.__save_history(result)

        response = self.__extract_answer(result)
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"pipeline": self.ppl}
