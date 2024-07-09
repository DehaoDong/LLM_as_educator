from codellama.llama import Llama


class ModelHandler:
    def __init__(self,
                 model='CodeLlama-7b-Instruct',
                 temperature=0.2,
                 top_p=0.95,
                 max_seq_len=512,
                 max_batch_size=8,
                 max_gen_len=None):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.max_gen_len = max_gen_len

        self.ckpt_dir = f'codellama/{model}/'
        self.tokenizer_path = f'codellama/{model}/tokenizer.model'

        self.generator = Llama.build(
            ckpt_dir=self.ckpt_dir,
            tokenizer_path=self.tokenizer_path,
            max_seq_len=self.max_seq_len,
            max_batch_size=self.max_batch_size,
        )

    def generate_response(self, instructions):
        results = self.generator.chat_completion(
            instructions,
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        response = results[0]['generation']['content'].strip()
        return response
