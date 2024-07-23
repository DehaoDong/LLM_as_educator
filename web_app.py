import json
import torch
from flask import Flask, request, jsonify, render_template
from model import CodeLlama
import knowledge_base as kb
from langchain.chains import RetrievalQA
import prompt_engineering as pe
from werkzeug.utils import secure_filename
import os
from transformers import pipeline

# Initialize Flask application
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['UPLOAD_FOLDER'] = 'documents'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'doc', 'docx'}
app.config['FINE_TUNE_DATASET'] = 'fine_tuning/datasets/fine_tune_dataset.json'

llm = CodeLlama(pipeline=pipeline(task="text-generation",
                                  model="meta-llama/CodeLlama-7b-Instruct-hf",
                                  max_new_tokens=512,
                                  device_map="auto",
                                  torch_dtype=torch.bfloat16))

# Define the Flask route for processing prompts
@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    retriever = kb.get_knowledge_base_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": pe.QA_PROMPT_TEMPLATE}
    )

    response = qa.invoke(prompt)  # Use the LLM instance directly as a callable
    response_content = response['result']

    web_response = {
        "response": response_content,
    }

    return jsonify(web_response)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'documents' not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist('documents')
    if not files:
        return jsonify({"error": "No selected files"}), 400

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else:
            return jsonify({"error": f"File type not allowed for file {file.filename}"}), 400

    kb.build_knowledge_base()

    return jsonify({"message": "Files uploaded and knowledge base updated successfully."})


@app.route('/history', methods=['GET'])
def get_history():
    history_file = 'history/history.json'
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
        return jsonify(history)
    else:
        return jsonify([])


@app.route('/fine-tune-dataset', methods=['GET'])
def get_fine_tune_dataset():
    if os.path.exists(app.config['FINE_TUNE_DATASET']):
        with open(app.config['FINE_TUNE_DATASET'], 'r') as f:
            fine_tune_dataset = json.load(f)
        return jsonify(fine_tune_dataset)
    else:
        return jsonify([])


@app.route('/save-finetune', methods=['POST'])
def save_fine_tune():
    fine_tune_data = request.get_json()
    with open(app.config['FINE_TUNE_DATASET'], 'w') as f:
        json.dump(fine_tune_data, f, indent=2)
    return jsonify({"message": "Fine-tuning data saved successfully."})


# Serve the frontend
@app.route('/')
def serve_index():
    return render_template('index.html')

@app.route('/monitor')
def serve_monitor():
    return render_template('monitor.html')


# Start the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
