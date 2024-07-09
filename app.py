from flask import Flask, request, jsonify, render_template
from typing import Optional
import time
from codellama.llama import Llama

# Initialize Flask application
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Load the model
model = 'CodeLlama-7b-Instruct'
temperature = 0.2
top_p = 0.95
max_seq_len = 512
max_batch_size = 8
max_gen_len = None

ckpt_dir = f'codellama/{model}/'
tokenizer_path = f'codellama/{model}/tokenizer.model'

generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
)


# Function to handle model inference
def generate_response(prompt):
    instructions = [
        [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    ]

    results = generator.chat_completion(
        instructions,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    response = results[0]['generation']['content']

    response = response.strip()

    print(f'response: {response}')

    return response


# Define the Flask route for processing prompts
@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Call the generate_response function
    response_content = generate_response(prompt)

    response = {
        "response": response_content,
    }

    return jsonify(response)


# Serve the frontend
@app.route('/')
def serve_frontend():
    return render_template('index.html')


# Start the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
