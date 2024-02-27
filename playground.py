from inference import InferFromModel
from transformers import AutoTokenizer

# inference
checkpoint_thesa = "johnhandleyd/thesa"
tokenizer_thesa = AutoTokenizer.from_pretrained(checkpoint_thesa)

example = "I've been feeling depressed lately. Can you help me?"
prompt = "You are a therapist helping a patient with their depression. Be kind and brief."

sample = InferFromModel(model=checkpoint_thesa, tokenizer=tokenizer_thesa)
print(sample.infer(example=example, prompt=prompt))