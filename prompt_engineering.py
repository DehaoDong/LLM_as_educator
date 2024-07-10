from typing import Dict, List

class PromptBuilder:
    context: List[Dict[str, str]]
    context_window: int

    def __init__(self, context_window: int = 2):
        self.context = []
        self.context_window = context_window

    def build_instruction(self, prompt: str) -> List[List[Dict[str, str]]]:
        context_str = "".join(
            # f"user: {interaction['user']} assistant: {interaction['assistant']} "
            interaction['user']
            for interaction in self.context
        )

        instructions = [
            [
                {
                    "role": "system",
                    "content": "Below are some historical messages, use them as context."
                               + context_str,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        ]

        print(instructions)

        return instructions

    def save_context(self, interaction: Dict[str, str]) -> None:
        if len(self.context) >= self.context_window:
            self.context.pop(0)
        self.context.append(interaction)