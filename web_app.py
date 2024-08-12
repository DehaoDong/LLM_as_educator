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
is_finetuned = False

# LLM handler
ppl = get_model_pipeline(model, is_finetuned=False)
llm = CodeLlama(ppl=ppl)


# Generate a response
@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # filled_prompt = pe.BASE_PROMPT_TEMPLATE.format(prompt=prompt)

    filled_prompt = pe.QA_PROMPT_TEMPLATE.format(question=prompt)

    response_content = llm.invoke(filled_prompt)



    # retriever = kb.get_knowledge_base_retriever()
    #
    # qa = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     retriever=retriever,
    #     chain_type_kwargs={"prompt": pe.QA_PROMPT_TEMPLATE}
    # )

    # response = qa.invoke(prompt)
    # response_content = response['result']

    # assignment = prompt
    #
    # rubric = '''
    # Assignment: Investigating and Analysing Ethical Risks of a Digital Engagement application \n\nThis is an INDIVIDUAL assignment on Ethical Risk Assessment for the TEU00062 module. This assessment is worth 30% of the total TEU 00062 module mark. The deadline for submission is 12.00. noon Friday 31st March 2024. \n \nLearning outcomes: Upon completion of this assignment students should:\nBe able to identify the different stakeholders and organisational roles involved in developing and using AI-driven digital engagement application, and motivations. \nBe able to identify and analyse ethical risks in a digital engagement application.\nBe equipped to identify technical and governance mitigations of these risks\n \nSteps:\n \nSelect ONE real world application that embodies one or more AI-driven digital engagement techniques you have learnt about on this course, e.g. e.g. digital engagement techniques used in one applications such as social media, newsfeeds, search, eCommerce, elearning or digital humanities., or in an application you are familiar with e.g. interactive Game, Virtual Presence (i.e. VR /AR) etc. \nIdentify the different value chain stakeholder roles involved in this application scenario and where possible the actors (i.e. companies, organisations) carrying out those roles. including those involved in: data collection; AI model development; development and use of the application. \nUse social responsibility principles (see ethic lecture week 2) to identify and analyse real or potential ethical risks of the application, including classification of the affected stakeholders (hint: see also risks classified by Latzer et al 2016 \u2013 ethics lecture week 1), how severe the risk would be and how likely it is to occur.\nIdentify one or more mitigation measures that specific value-chain stakeholders may undertake to minimise the impact of the risk you think is most harmful. These could be technical measures, organisational governance measures or measure involving additional engagement with affected individuals or groups.\n \nStructure of Assignment Report. \nStructure the Report (4000 word max excluding figures and references), under the following heading [indicative percentage of assignment mark per section]. Please provide references\n \nDescription of Application [20%]\n<describe your selected application, explaining what it does, the role AI and data collection plays in its digital engagement features, and the benefits it provides>\n\nIdentification of Stakeholder Roles involved in the application and its governance [10%]\n<describe the stakeholder roles undertaken in the development, use and governance of the application, and which organization take those roles and the benefits they derive from their involvement. Include roles that may be indirectly affected by the application. >\n\nIdentification of Ethical Risks [40%]\n<Identify the ethical risks that may be raised by your application under the social responsibility categories below (more details in ethics lecture week 2). For each risk identify which stakeholder may be affected, and assess how severe the impact would be for each stakeholder (Low, Medium, High) and the likelihood of that risk being incurred (Low, Medium, High), justifying your assessment, even if you think there is no risk under that heading.>\nHuman Rights\nLabour Practices\nThe Environment\nFair Operating Procedures\nConsumer Issues\nCommunity Involvement and Development\n \nDiscussion of Mitigations Measures for Risk [20%]\n< For one of the risks you consider more severe, describe mitigation measures that could be taken. Explain how the measure would reduce or eliminate the likelihood and impact of the risk. Explain which stakeholders would be involved, or would have to interact in implementing the measure, identifying if this is a governance measure an organization can undertake itself or in collaboration with other parties, e.g. regulators or external stakeholder..>\n\nReferences [10%]\nInclude references to source you use for you analysis, which could be academic or policy papers, blogs, information from company web site or news stories.\n
    # '''
    #
    # filled_prompt = pe.FE_PROMPT_TEMPLATE.format(rubric=rubric, assignment=assignment)

    # response_content = llm.invoke(filled_prompt)

    web_response = {
        "response": response_content,
    }

    return jsonify(web_response)

@app.route('/change_model', methods=['POST'])
def change_model():
    global model, is_finetuned, ppl, llm

    data = request.get_json()
    selected_model = data.get('model', '')
    ft_flag = data.get('is_finetuned', False)

    # print(f'slected model: {selected_model}')
    # print(f'is_finetuned: {is_finetuned}')

    if selected_model and (selected_model != model or ft_flag != is_finetuned):
        del ppl, llm
        torch.cuda.empty_cache()

        model = selected_model
        is_finetuned = ft_flag
        ppl = get_model_pipeline(model, is_finetuned)
        llm = CodeLlama(ppl=ppl)
        return jsonify({"message": f"Model changed to {model} (Fine-tuned: {is_finetuned}) successfully."})
    else:
        return jsonify({"message": "No change in model or invalid model selected."}), 400


# Check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Upload files
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


# Get history
@app.route('/history', methods=['GET'])
def get_history():
    history_file = 'history/history.json'
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
        return jsonify(history)
    else:
        return jsonify([])


# Get fine-tune dataset
@app.route('/fine-tune-dataset', methods=['GET'])
def get_fine_tune_dataset():
    if os.path.exists(app.config['FINE_TUNE_DATASET']):
        with open(app.config['FINE_TUNE_DATASET'], 'r') as f:
            fine_tune_dataset = json.load(f)
        return jsonify(fine_tune_dataset)
    else:
        return jsonify([])


# Save fine-tune dataset
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
                return jsonify(
                    {"error": "Invalid dataset format. Each message must contain 'role' and 'content'."}), 400
        if not any(message['role'] == 'assistant' for message in conversation):
            return jsonify({"error": "Each conversation must include an 'assistant' message."}), 400

    # Ensure the dataset aligns with the structure including system prompts
    with open(app.config['FINE_TUNE_DATASET'], 'w') as f:
        json.dump(fine_tune_data, f, indent=2)

    # offload model
    global ppl, llm
    del ppl, llm
    torch.cuda.empty_cache()

    # Trigger fine-tuning
    success, message = fine_tune(model)
    torch.cuda.empty_cache()
    if success:
        # reload model
        ppl = get_model_pipeline(model)
        llm = CodeLlama(ppl=ppl)

        return jsonify({"message": message}), 200
    else:
        # reload model
        ppl = get_model_pipeline(model)
        llm = CodeLlama(ppl=ppl)

        return jsonify({"error": message}), 500


# Augment a conversation
@app.route('/augment-finetune', methods=['POST'])
def augment_fine_tune():
    data = request.get_json()

    if not isinstance(data, list) or len(data) != 3:
        return jsonify(
            {"error": "Invalid input data. Expected a list of three messages: system, user, and assistant."}), 400

    system_message = next((msg for msg in data if msg['role'] == 'system'), None)
    user_message = next((msg for msg in data if msg['role'] == 'user'), None)
    assistant_message = next((msg for msg in data if msg['role'] == 'assistant'), None)

    if not system_message or not user_message or not assistant_message:
        return jsonify({"error": "Invalid input data. Each message must contain 'role' and 'content' fields."}), 400

    try:
        # Format the prompt for augmentation
        arguments = {
            "prompt": user_message['content'],
            "response": assistant_message['content']
        }
        ad_prompt = pe.DA_PROMPT_TEMPLATE.format(**arguments)

        # Use llm to generate 4 more user and assistant pairs
        response = llm.invoke(ad_prompt)
        # print(f'Response: {response}')

        augmented_pairs = json.loads(response)
        # print(f'Augmented pairs: {augmented_pairs}')

        if not isinstance(augmented_pairs, list):
            return jsonify({"error": "Invalid model response format. Expected a list of pairs."}), 500

        new_records = []
        for pair in augmented_pairs:
            # print(f'Pair: {pair}')
            new_record = [
                system_message,
                {"role": "user", "content": pair[0]['content']},
                {"role": "assistant", "content": pair[1]['content']}
            ]
            new_records.append(new_record)

        # print(f'New records: {new_records}')

        return jsonify({"message": "Fine-tuning dataset augmented successfully.", "new_records": new_records}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Serve the frontend
@app.route('/')
def serve_index():
    return render_template('index.html')


# Monitor page
@app.route('/monitor')
def serve_monitor():
    return render_template('monitor.html')


# Start the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
