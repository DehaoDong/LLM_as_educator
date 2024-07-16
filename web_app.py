from flask import Flask, request, jsonify, render_template
from model import CodeLlama, ModelHandler
import knowledge_base as kb
from langchain.chains import RetrievalQA
import prompt_engineering as pe
from werkzeug.utils import secure_filename
import os

# Initialize Flask application
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['UPLOAD_FOLDER'] = 'documents'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'doc', 'docx'}

llm = CodeLlama(model_handler=ModelHandler(model='CodeLlama-7b-Instruct'))


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
        chain_type_kwargs={"prompt": pe.QA_CHAIN_PROMPT}
    )

    response = qa.invoke(prompt)  # Use the LLM instance directly as a callable

    print(response)

    response = {
        "response": response['result'],
    }

    return jsonify(response)


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


# Serve the frontend
@app.route('/')
def serve_frontend():
    return render_template('index.html')


# Start the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
