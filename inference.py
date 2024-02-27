import torch, locale
from cleaning import CleanData
from peft import AutoPeftModelForCausalLM
from transformers import GenerationConfig, AutoTokenizer

locale.getpreferredencoding = lambda: "UTF-8"

class InferFromModel():
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


    def normalise_example(self, sample_text, prompt=None):
        """
        Apply the chat template to the example using the CleanData().apply_template() method from 'cleaning.py'.
        """
        template = {"question": sample_text,
                    "answer": ""}
        normalised_ex = CleanData().apply_template(template, prompt_message=prompt)
        
        return normalised_ex['templateText']


    def infer(self, example, prompt: str = None, cuda: bool = True, n_examples: int = 1, temp: float = 0.1, max_tokens: int = 256):
        """
        Do inference with the fine-tuned model.
        """
        model = AutoPeftModelForCausalLM.from_pretrained(
                    self.model,
                    low_cpu_mem_usage=True,
                    return_dict=True,
                    torch_dtype=torch.float16,
                    device_map="cuda")
        
        example = self.normalise_example(example, prompt=prompt)

        if cuda:
            inputs = self.tokenizer(example, return_tensors="pt").to("cuda")
        else:
            inputs = self.tokenizer(example, return_tensors="pt")

        config = GenerationConfig(
            do_sample=True,
            top_k=n_examples,
            temperature=temp,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id
            )
        
        output = model.generate(**inputs, generation_config=config)
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        result = "<|user|>".join(result.split("<|user|>")[:2])  # Show only the first interchange, sometimes it brings a new one
        
        return result.replace(r'\n', '\n')
        

