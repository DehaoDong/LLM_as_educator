def build_instruction(prompt):
    instructions = [
        [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    ]

    return instructions