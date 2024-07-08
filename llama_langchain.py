from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import datetime
import time
import threading
from llama import Llama

class CodeLlamaLLM(LLM):
    def __init__(self, ckpt_dir: str, tokenizer_path: str, temperature: float, top_p: float, max_seq_len: int, max_batch_size: int, max_gen_len: Optional[int]):
        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )
        self.temperature = temperature
        self.top_p = top_p
        self.max_gen_len = max_gen_len

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        instructions = [
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        ]

        results = self.generator.chat_completion(
            instructions,  # type: ignore
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        response = results[0]['generation']['content']
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": "CodeLlama"}

    @property
    def _llm_type(self) -> str:
        return "code_llama_llm"


from langchain import PromptTemplate
from langchain.chains import LLMChain

# Define your CodeLlama parameters
ckpt_dir = 'CodeLlama-7b-Instruct/'
tokenizer_path = 'CodeLlama-7b-Instruct/tokenizer.model'
temperature = 0.2
top_p = 0.95
max_seq_len = 512
max_batch_size = 8
max_gen_len = 128  # You can adjust this as needed

# Initialize the CodeLlama LLM
code_llama_llm = CodeLlamaLLM(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    temperature=temperature,
    top_p=top_p,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
    max_gen_len=max_gen_len
)

# Define a prompt template
template = "Translate the following English text to French: {text}"
prompt = PromptTemplate(template=template, input_variables=["text"])

# Create an LLMChain with the CodeLlama LLM and prompt template
llm_chain = LLMChain(prompt=prompt, llm=code_llama_llm)

# Run the chain with some input
result = llm_chain.run(text="Hello, how are you?")
print(result)
