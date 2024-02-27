from cleaning import CleanData
from finetuning import FineTune
from inference import InferFromModel
from transformers import AutoTokenizer

# get and clean dataset
thesa_dataset = CleanData().get_data()

# import Zephyr 7B model and tokenizer
checkpoint_zephyr = "TheBloke/zephyr-7B-alpha-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_zephyr)

# instantiate model (parameters in finetuning.py)
finetuned_model = FineTune(
    dataset=thesa_dataset,
    checkpoint=checkpoint_zephyr,
    tokenizer=tokenizer
            )

# finetune model
finetuned_model.finetune(output="/thesa",
                         epochs=10)

# inference
checkpoint_thesa = "johnhandleyd/thesa"
tokenizer_thesa = AutoTokenizer.from_pretrained(checkpoint_thesa)

example = "I've been feeling depressed lately. Can you help me?"

sample = InferFromModel(model=checkpoint_thesa, tokenizer=tokenizer_thesa)
print(sample.infer(example=example))