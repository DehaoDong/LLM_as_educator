import json
from datasets import load_dataset
import os

# Load the SQuAD v2 dataset
ds = load_dataset("rajpurkar/squad_v2")

# Dictionary to store unique article contents by title
articles = {}

# Set to store unique question-answer pairs
qa_pairs_set = set()

for row in ds["validation"]:
    title = row['title']
    context = row['context']

    if title not in articles:
        articles[title] = set()  # Use a set to store unique contexts

    articles[title].add(context)  # Add context to the set to ensure uniqueness

    question = row['question']
    answer_texts = row['answers']['text']

    # Add each question-answer pair to the set for uniqueness
    for answer in answer_texts:
        qa_pairs_set.add((question, answer))

# Convert the set of tuples back to a list of dictionaries
qa_pairs = [{"question": question, "answer": answer} for question, answer in qa_pairs_set]

# Create a directory to save the output files
output_dir = '../dataset/squad_v2/articles'
os.makedirs(output_dir, exist_ok=True)

# Write each article's contents to a separate text file
for title, contexts in articles.items():
    # Sanitize title for use as filename
    sanitized_title = title.replace("/", "-").replace("\\", "-").replace(" ", "_")
    file_path = os.path.join(output_dir, f"{sanitized_title}.txt")

    print(f'writing {file_path}')

    with open(file_path, 'w', encoding='utf-8') as file:
        for paragraph in sorted(contexts):  # Sorting contexts to maintain order
            file.write(paragraph + "\n\n")  # Separate paragraphs with new lines

print(f"Articles saved to: {output_dir}")

# Save the unique question-answer pairs to a JSON file
output_file = '../dataset/squad_v2/qa_pairs.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(qa_pairs, f, ensure_ascii=False, indent=4)

print(f"Question-answer pairs saved to: {output_file}")
