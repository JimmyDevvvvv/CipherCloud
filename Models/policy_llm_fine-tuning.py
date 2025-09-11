import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import os

class PolicyLLMTrainer:
    def __init__(self, model_repo: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        self.model_repo = model_repo
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("Configuring 4-bit quantization with GPU support and model streaming...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        print("Loading tokenizer with streaming...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_repo,
            trust_remote_code=True,
            token=True  # Use Hugging Face token if required
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Loading base model with 4-bit quantization and streaming...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            load_in_8bit_fp32_cpu_offload=False,  # Disable CPU offload for GPU focus
            streaming=True  # Enable streaming
        )
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        
        print("Applying LoRA adapters...")
        self.model = get_peft_model(self.model, self.lora_config)
        
        trainable_params = 0
        all_params = 0
        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"✨ Model prepared with QLoRA using streaming!")
        print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}% of {all_params:,} total)")

    def prepare_training_data(self, dataset_path: str = "Binary Dataset/cipher_cloud_dataset.json"):
        print("Loading and preparing dataset...")
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} policies from dataset")
        
        # Limit to 100 examples for initial run
        data = data[:100]
        print(f"Using {len(data)} policies for training (limited to 100)")
        
        train_data = []
        for item in data:
            prompt = f"Given this IAM policy, describe a potential attack:\n{json.dumps(item['policy'], indent=2)}\nOutput:"
            completion = f"\n{item['attack']}"
            train_data.append({"prompt": prompt, "completion": completion})
        
        dataset = Dataset.from_list(train_data)
        
        def tokenize_function(examples):
            print(f"Tokenizing batch of {len(examples['prompt'])} examples...")
            tokenized_inputs = self.tokenizer(
                examples["prompt"],
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )
            tokenized_targets = self.tokenizer(
                examples["completion"],
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )
            
            input_ids = []
            labels = []
            attention_mask = []
            for i in range(len(examples["prompt"])):
                combined_input_ids = torch.cat([tokenized_inputs["input_ids"][i], tokenized_targets["input_ids"][i]])
                combined_labels = torch.cat([torch.full_like(tokenized_inputs["input_ids"][i], -100), tokenized_targets["input_ids"][i]])
                combined_attention_mask = torch.cat([tokenized_inputs["attention_mask"][i], tokenized_targets["attention_mask"][i]])
                if combined_input_ids.size(0) > 512:
                    combined_input_ids = combined_input_ids[:512]
                    combined_labels = combined_labels[:512]
                    combined_attention_mask = combined_attention_mask[:512]
                input_ids.append(combined_input_ids)
                labels.append(combined_labels)
                attention_mask.append(combined_attention_mask)
            
            return {
                "input_ids": torch.stack(input_ids),
                "attention_mask": torch.stack(attention_mask),
                "labels": torch.stack(labels)
            }
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            load_from_cache_file=False
        )
        return tokenized_dataset

    def train(self, dataset, output_dir: str = "Models/policy_llm_finetuned"):
        print(f"\nTraining Configuration:")
        print(f"Batch size: 1")
        print(f"Gradient accumulation steps: 16")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=5e-5,
            fp16=True,
            logging_steps=1,
            save_strategy="epoch",
            warmup_steps=50,
            lr_scheduler_type="linear",
            report_to="none",
            gradient_checkpointing=True,
            optim="adamw_torch",
            max_grad_norm=0.3,
            remove_unused_columns=False,
            group_by_length=True,
            dataloader_pin_memory=True,
            dataloader_num_workers=2,
            do_eval=False,
            save_total_limit=1,
        )

        def data_collator(features):
            def to_tensor(x):
                if isinstance(x, torch.Tensor):
                    return x
                return torch.tensor(x)
            return {
                'input_ids': torch.stack([to_tensor(f['input_ids']) for f in features]),
                'attention_mask': torch.stack([to_tensor(f['attention_mask']) for f in features]),
                'labels': torch.stack([to_tensor(f['labels']) for f in features])
            }

        print(f"\nDataset statistics:")
        print(f"Number of examples: {len(dataset)}")
        if len(dataset) == 0:
            raise ValueError("Dataset is empty!")
            
        first_example = dataset[0]
        print("\nFirst example contents:")
        for key, value in first_example.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
            else:
                print(f"{key}: {type(value)}")

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )

        print("Starting training...")
        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(workspace_dir, "Binary Dataset", "cipher_cloud_dataset.json")
    output_dir = os.path.join(workspace_dir, "Models", "policy_llm_finetuned")
    
    print(f"\nWorkspace directory: {workspace_dir}")
    print(f"Looking for dataset at: {dataset_path}")
    print(f"Model will be saved to: {output_dir}")
    
    trainer = PolicyLLMTrainer()
    dataset = trainer.prepare_training_data()
    trainer.train(dataset, output_dir=output_dir)
    
    print("✨ Fine-tuning complete!")

if __name__ == "__main__":
    main()
