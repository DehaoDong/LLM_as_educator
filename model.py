from typing import Any, List, Mapping, Optional, Dict
from langchain.llms.base import LLM
from codellama.llama import Llama
import prompt_engineering as pe


class ModelHandler:
    def __init__(self,
                 model='CodeLlama-7b-Instruct',
                 temperature=0.2,
                 top_p=0.95,
                 max_seq_len=1024,
                 max_batch_size=8,
                 max_gen_len=None):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.max_gen_len = max_gen_len

        self.ckpt_dir = f'codellama/{model}/'
        self.tokenizer_path = f'codellama/{model}/tokenizer.model'

        self.generator = Llama.build(
            ckpt_dir=self.ckpt_dir,
            tokenizer_path=self.tokenizer_path,
            max_seq_len=self.max_seq_len,
            max_batch_size=self.max_batch_size,
        )

    def build_instruction(slef, prompt: str) -> List[List[Dict[str, str]]]:
        instructions = [
            [
                {
                    "role": "system",
                    "content": pe.SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        ]

        return instructions

    def generate(self, prompt):
        instructions = self.build_instruction(prompt)

        print(instructions)

        results = self.generator.chat_completion(
            instructions,
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        response = results[0]['generation']['content'].strip()
        return response

    def __str__(self):
        return f"{self.model}"


class CodeLlama(LLM):
    model_handler: ModelHandler

    @property
    def _llm_type(self) -> str:
        return str(self.model_handler)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.model_handler.generate(prompt)
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_handler": str(self.model_handler)}

