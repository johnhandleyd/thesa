import wandb, os
from huggingface_hub import login
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, TrainingArguments, EarlyStoppingCallback
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

# HuggingFace login
login()

# Wandb login
wandb.login()
os.environ["WANDB_PROJECT"]="thesa_v1"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

class FineTune():

    def __init__(self, dataset: Dataset, checkpoint, tokenizer: AutoTokenizer):
        self.dataset = dataset
        self.checkpoint = checkpoint
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    
    def prepare_model(self):
        gptq_config = GPTQConfig(bits=4,
                    disable_exllama=True,
                    lora_r=16,
                    lora_alpha=16,
                    tokenizer=self.tokenizer
                    )

        model = AutoModelForCausalLM.from_pretrained(self.checkpoint,
                                              quantization_config=gptq_config,
                                              device_map="auto",
                                              use_cache=False,
                                              )
        model.config.pretraining_tp = 1
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
                    r=16,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj", "v_proj"]
                    )
        
        model = get_peft_model(model, peft_config)
        
        return model, peft_config


    def finetune(
            self,
            output_dir: str = "thesa",
            batch_size: int = 8,
            lr: float = 2e-4,
            save_every: str = "epoch",
            epochs: int = 2,
            push_to_hub: str = True,
            max_seq_length: int = 512
        ):
        """
        Fine-tune model. Currently stops following max steps.
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            optim="paged_adamw_32bit",
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            save_strategy=save_every,
            logging_steps=100,
            num_train_epochs=epochs,
            fp16=True,
            report_to="wandb",
            run_name="thesa_v1_lr2e-4",
            push_to_hub=push_to_hub)
         
        self.tokenizer.padding_side = 'right'
        model, peft_config = self.prepare_model()
        # early_stopping = EarlyStoppingCallback(5)

        trainer = SFTTrainer(
            model=model,
            train_dataset=self.dataset.shuffle(seed=35),
            peft_config=peft_config,
            dataset_text_field="templateText",
            args=training_args,
            tokenizer=self.tokenizer,
            packing=False,
            max_seq_length=max_seq_length
            # callbacks=[early_stopping]
        )
        
        trainer.train()

        if push_to_hub == True:
            trainer.push_to_hub()

