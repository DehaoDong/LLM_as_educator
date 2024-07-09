# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional
import fire
from codellama.llama import Llama
import datetime
import time
import threading


def main(
        ckpt_dir: str = 'codellama/CodeLlama-7b-Instruct/',
        tokenizer_path: str = 'codellama/CodeLlama-7b-Instruct/tokenizer.model',
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_seq_len: int = 512,
        max_batch_size: int = 8,
        max_gen_len: Optional[int] = None,
):
    # Initialize the Llama model
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    def display_thinking_time(start_time):
        while not stop_thread.is_set():
            elapsed_time = time.time() - start_time
            print(f"\r> CodeLlama is thinking... {elapsed_time:.0f}s", end="")
            time.sleep(1)

    while True:
        prompt = input(f"> User({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}): ")
        instructions = [
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        ]

        print("CodeLlama is thinking...", end="")

        start_time = time.time()
        stop_thread = threading.Event()
        thinking_thread = threading.Thread(target=display_thinking_time, args=(start_time,))
        thinking_thread.start()

        results = generator.chat_completion(
            instructions,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        stop_thread.set()
        thinking_thread.join()
        print()  # Print newline to end the thinking time display

        for result in results:
            print(
                f"> CodeLlama({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}): {result['generation']['content']}")
            print("\n=============================================================\n")


if __name__ == "__main__":
    fire.Fire(main)
