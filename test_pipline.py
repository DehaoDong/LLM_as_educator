from fine_tune import fine_tune
from model import get_model_pipeline, CodeLlama

prompt = '''
<<SYS>>
<</SYS>>
<<USR>>
who are you?
<</USR>>
'''

model = "CodeLlama-7b-Instruct-hf"

success, message = fine_tune(model, learning_rate=3e-4, num_train_epochs=30)
print(success, message)

ppl = get_model_pipeline(model)
llm = CodeLlama(ppl=ppl)

response = llm.invoke(prompt)

print(response)
