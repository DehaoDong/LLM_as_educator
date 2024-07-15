from typing import List, Dict


def build_instruction(prompt: str) -> List[List[Dict[str, str]]]:
    instructions = [
        [
            {
                "role": "system",
                "content": "You are a professional educator. You are going to provide guidance and explanation to a "
                           "student to help them understand concepts or solve problems.",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
    ]

    print(instructions)

    return instructions
