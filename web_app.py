import torch
from flask import Flask, request, jsonify, render_template
from fine_tune import fine_tune
from model import CodeLlama, get_model_pipeline
import knowledge_base as kb
from langchain.chains import RetrievalQA
import prompt_engineering as pe
from werkzeug.utils import secure_filename
import os
import json

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['UPLOAD_FOLDER'] = 'documents'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'doc', 'docx'}
app.config['FINE_TUNE_DATASET'] = 'fine_tuning/datasets/fine_tune_dataset.json'

model = "CodeLlama-7b-Instruct-hf"

ppl = get_model_pipeline(model)
llm = CodeLlama(ppl=ppl)

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

    response = qa.invoke(prompt)
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

    # Validate the dataset
    for conversation in fine_tune_data:
        if not isinstance(conversation, list):
            return jsonify({"error": "Invalid dataset format. Each conversation must be a list of messages."}), 400
        for message in conversation:
            if not isinstance(message, dict):
                return jsonify({"error": "Invalid dataset format. Each message must be a dictionary."}), 400
            if not all(key in message for key in ['role', 'content']):
                return jsonify({"error": "Invalid dataset format. Each message must contain 'role' and 'content'."}), 400
        if not any(message['role'] == 'assistant' for message in conversation):
            return jsonify({"error": "Each conversation must include an 'assistant' message."}), 400

    # Ensure the dataset aligns with the structure including system prompts
    with open(app.config['FINE_TUNE_DATASET'], 'w') as f:
        json.dump(fine_tune_data, f, indent=2)

    # Trigger fine-tuning
    success, message = fine_tune(model)
    if success:
        # reload model
        global ppl, llm

        del llm
        del ppl

        torch.cuda.empty_cache()

        ppl = get_model_pipeline(model)
        llm = CodeLlama(ppl=ppl)

        return jsonify({"message": message}), 200
    else:
        return jsonify({"error": message}), 500

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
