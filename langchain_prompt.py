from langchain.llms.base import LLM
from typing import Optional, List
from codellama.llama import Llama
from pydantic import BaseModel, Field

class CodeLlamaLLM(LLM, BaseModel):
    ckpt_dir: str
    tokenizer_path: str
    max_seq_len: int
    max_batch_size: int
    temperature: float
    top_p: float
    max_gen_len: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generator = Llama.build(
            ckpt_dir=self.ckpt_dir,
            tokenizer_path=self.tokenizer_path,
            max_seq_len=self.max_seq_len,
            max_batch_size=self.max_batch_size,
        )

    def _call(self, prompt: str, stop: Optional[List[str]] = None, max_tokens: Optional[int] = None):
        instructions = [[{"role": "user", "content": prompt}]]

        results = self.generator.chat_completion(
            instructions,  # type: ignore
            max_gen_len=self.max_gen_len if max_tokens is None else max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        # Assuming you want the first result
        return results[0]['generation']['content']

    def _identifying_params(self):
        return {
            "ckpt_dir": self.ckpt_dir,
            "tokenizer_path": self.tokenizer_path,
            "max_seq_len": self.max_seq_len,
            "max_batch_size": self.max_batch_size,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_gen_len": self.max_gen_len,
        }

    @property
    def _llm_type(self) -> str:
        return "custom_codellama"



from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize the custom LLM
code_llama_llm = CodeLlamaLLM(
    ckpt_dir='codellama/CodeLlama-7b-Instruct/',
    tokenizer_path='codellama/CodeLlama-7b-Instruct/tokenizer.model',
    temperature=0.2,
    top_p=0.95,
    max_seq_len=512,
    max_batch_size=8,
    max_gen_len=100,  # You can adjust this value as needed
)

# Define a simple prompt
prompt = PromptTemplate(input_variables=["question"], template="Q: {question}\nA:")

# Create an LLMChain
llm_chain = LLMChain(llm=code_llama_llm, prompt=prompt)

# Use the chain to generate a response
question = "What is the capital of France?"
response = llm_chain.run(question)
print(response)
