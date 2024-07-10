from typing import Optional, List, Dict
import fire
from codellama.llama import Llama
import datetime
import time
import threading


def add_context(
        new_instruction: Dict[str, str],
        past_interactions: List[Dict[str, str]],
        context_length: int = 5
) -> List[Dict[str, str]]:
    """
    Add context to the new instruction from past interactions.

    Args:
    - new_instruction (Dict[str, str]): The new instruction to be processed.
    - past_interactions (List[Dict[str, str]]): The list of past interactions.
    - context_length (int): The number of past interactions to include as context.

    Returns:
    - List[Dict[str, str]]: The updated instructions with context.
    """
    past_interactions.append(new_instruction)

    # Limit the number of interactions to the context length
    if len(past_interactions) > context_length:
        past_interactions = past_interactions[-context_length:]

    return past_interactions


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

    # List to hold past interactions
    past_interactions = []

    while True:
        prompt = input(f"> User({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}): ")

        # Add the new prompt to past interactions
        new_instruction = {
            "role": "user",
            "content": prompt,
        }
        past_interactions = add_context(new_instruction, past_interactions)

        # Prepare instructions with context
        instructions = [past_interactions]

        print("CodeLlama is thinking...", end="")

        start_time = time.time()
        stop_thread = threading.Event()
        thinking_thread = threading.Thread(target=display_thinking_time, args=(start_time,))
        thinking_thread.start()

        print(instructions)

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
            assistant_reply = {
                "role": "assistant",
                "content": result['generation']['content']
            }
            past_interactions = add_context(assistant_reply, past_interactions)
            print(
                f"> CodeLlama({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}): {result['generation']['content']}")
            print("\n=============================================================\n")


if __name__ == "__main__":
    fire.Fire(main)
