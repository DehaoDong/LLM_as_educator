from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any, Dict
import datetime
import time
import threading
from codellama.llama import Llama
import fire


class CustomLLM(LLM):
    def __init__(self, ckpt_dir: str, tokenizer_path: str, temperature: float, top_p: float, max_seq_len: int,
                 max_batch_size: int):
        super().__init__()
        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )
        self.temperature = temperature
        self.top_p = top_p

    @property
    def _llm_type(self) -> str:
        return "custom_llama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted")

        instructions = [
            [
                {
                    "role": "system",
                    "content": "There are some historical interactions between you and user, use them as context to answer the following questions. "
                               + "user: my name is dehao"
                               + "assistant: hello",
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        ]

        results = self.generator.chat_completion(
            instructions,  # type: ignore
            max_gen_len=self.max_seq_len,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        # Assuming you want the content of the first result
        return results[0]['generation']['content'] if results else ""


def main(
        ckpt_dir: str = 'codellama/CodeLlama-7b-Instruct/',
        tokenizer_path: str = 'codellama/CodeLlama-7b-Instruct/tokenizer.model',
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_seq_len: int = 512,
        max_batch_size: int = 8,
):
    # Initialize the Custom LLM model
    llm = CustomLLM(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        temperature=temperature,
        top_p=top_p,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size
    )

    def display_thinking_time(start_time):
        while not stop_thread.is_set():
            elapsed_time = time.time() - start_time
            print(f"\r> CodeLlama is thinking... {elapsed_time:.0f}s", end="")
            time.sleep(1)

    while True:
        prompt = input(f"> User({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}): ")

        print("CodeLlama is thinking...", end="")

        start_time = time.time()
        stop_thread = threading.Event()
        thinking_thread = threading.Thread(target=display_thinking_time, args=(start_time,))
        thinking_thread.start()

        response = llm(prompt)

        stop_thread.set()
        thinking_thread.join()
        print()  # Print newline to end the thinking time display

        print(f"> CodeLlama({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}): {response}")
        print("\n=============================================================\n")


if __name__ == "__main__":
    fire.Fire(main)
