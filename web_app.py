from flask import Flask, request, jsonify, render_template
from model_handler import ModelHandler
from prompt_engineering import build_instruction

# Initialize Flask application
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Initialize the model handler
model_handler = ModelHandler(model='CodeLlama-7b-Instruct')

# Define the Flask route for processing prompts
@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    instructions = build_instruction(prompt)

    # Call the generate_response function from model_handler
    response_content = model_handler.generate_response(instructions)

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
