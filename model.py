import json
import os
from traceback import print_exc
from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM
from concurrent.futures import ThreadPoolExecutor

HISTORY_FILE = 'history/history.json'
HISTORY_LIMIT = 20

executor = ThreadPoolExecutor(max_workers=1)

class CodeLlama(LLM):
    pipeline: Any

    @property
    def _llm_type(self) -> str:
        return str(self.pipeline)

    def __extract_answer(self, result):
        # Access the first item in the 'generated_text' list
        generated_text = result[0]['generated_text']

        # Loop through the list to find the dictionary with the role 'assistant'
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

            # Append new result
            history.append(result[0]['generated_text'])

            # Limit history to the most recent 20 entries
            if len(history) > HISTORY_LIMIT:
                history = history[-HISTORY_LIMIT:]

            # Save updated history back to the file
            with open(HISTORY_FILE, 'w') as f:
                json.dump(history, f, indent=2)

            print(f'curent history: {len(history)}')

        # Run the save_to_file function in a separate thread
        executor.submit(save_to_file, result)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        sys_prompt_start = prompt.find("<SYS>") + len("<SYS>")
        sys_prompt_end = prompt.find("</SYS>")
        usr_prompt_start = prompt.find("<USR>") + len("<USR>")
        usr_prompt_end = prompt.find("</USR>")

        system_prompt = prompt[sys_prompt_start:sys_prompt_end].strip()
        user_prompt = prompt[usr_prompt_start:usr_prompt_end].strip()

        instruction = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        result = self.pipeline(instruction)
        print(json.dumps(result, indent=2))

        self.__save_history(result)

        response = self.__extract_answer(result)
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"pipeline": self.pipeline}

