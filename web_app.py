from flask import Flask, request, jsonify, render_template
from model import CodeLlama, ModelHandler

# Initialize Flask application
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

llm = CodeLlama(model_handler=ModelHandler(model='CodeLlama-7b-Instruct'))

# Define the Flask route for processing prompts
@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    response_content = llm.invoke(prompt)  # Use the LLM instance directly as a callable

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
