from copy import deepcopy
import pandas as pd
import json

# Load the dataset
file_path = '../dataset/filtered_ielts_writing_dataset_v2.csv'
data = pd.read_csv(file_path)

# # Select only the required columns
# filtered_data = data[['Question', 'Essay', 'Examiner_Commen', 'Overall']]
#
# # Save the filtered data to a new CSV file
# output_file_path = 'dataset/filtered_ielts_writing_dataset_v2.csv'
# filtered_data.to_csv(output_file_path, index=False)

# Define the template
template = [
    {
        "role": "system",
        "content": "You are a professional examiner who assesses students' essays based on the questions.\nThe question will be provided within <question> and </question> labels.\nThe essay content will be provided within <essay> and </essay> labels, make sure you identify the essay correctly.\nOutput follow a json format:\n{\n    \"Comment\": \"\",\n    \"Task_Response\": ,\n    \"Coherence_Cohesion\": ,\n    \"Lexical_Resource\": ,\n    \"Range_Accuracy\": ,\n    \"Overall\": \n}"
    },
    {
        "role": "user",
        "content": "<question>\n{question}\n</question>\n<essay>\n{essay}\n</essay>"
    },
    {
        "role": "assistant",
        "content": "{\n    \"Comment\": \"{Examiner_Commen}\",\n    \"Overall\": {Overall}\n}"
    }
]

# Fill NaN values with empty strings
data.fillna('', inplace=True)

# Build the JSON dataset
json_dataset = []

for _, row in data.iterrows():
    filled_template = deepcopy(template)

    # Fill in the user prompt with the specific question and essay from the dataset
    filled_template[1]['content'] = filled_template[1]['content'].format(
        question=row['Question'],
        essay=row['Essay']
    )

    # Fill in the assistant's response with the specific feedback and scores from the dataset
    filled_template[2]['content'] = filled_template[2]['content'].replace(
        "{Examiner_Commen}", row['Examiner_Commen']
    ).replace(
        "{Overall}", str(row['Overall'])
    )

    json_dataset.append(filled_template)

# Save the JSON dataset to a file
output_file_path = '../dataset/filled_json_dataset.json'
with open(output_file_path, 'w') as f:
    json.dump(json_dataset, f, indent=4)

print(f"JSON dataset saved to {output_file_path}")
