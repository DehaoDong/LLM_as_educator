from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM
import prompt_engineering as pe
from transformers import pipeline


class CodeLlama(LLM):
    pipeline: Any

    @property
    def _llm_type(self) -> str:
        return str(self.pipeline)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        instructions = [
            [
                # {
                #     "role": "system",
                #     "content": pe.SYSTEM_PROMPT
                # },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        ]
        result = self.pipeline(instructions)
        print(result)
        response = result[0][0]['generated_text'][1]['content'].strip()
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"pipeline": self.pipeline}

