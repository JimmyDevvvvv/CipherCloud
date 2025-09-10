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
import bitsandbytes as bnb
from typing import Dict, List
import os

class PolicyLLMTrainer:
    def __init__(self, model_path: str = "./model_cache/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize quantization config with CPU offloading
        print("Configuring 4-bit quantization with CPU offloading...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True
        )

        # Calculate optimal device map
        gpu_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        gpu_memory_gb = gpu_memory / (1024 ** 3)  # Convert to GB
        
        # Define base device map with all possible components
        base_device_map = {
            # Embeddings
            "model.embed_tokens": "cpu",
            "model.norm": "cpu",
            "lm_head": "cpu",
            
            # Transformer blocks
            **{f"model.layers.{i}": "cpu" for i in range(32)},  # Adjust range based on model size
            
            # Additional components
            "model.norm_f": "cpu",
            "base_model.model.norm": "cpu",
            "rotary_emb": "cpu",
            "gradient_checkpointing": "cpu",
            
            # DeepSeek specific components
            "model.wte": "cpu",
            "model.drop": "cpu",
            "model.h": "cpu",
            "model.ln_f": "cpu"
        }
        
        # Modify device map based on available GPU memory
        if gpu_memory_gb >= 24:  # High-end GPU
            device_map = "auto"
        elif gpu_memory_gb >= 8:  # Mid-range GPU
            # Move some layers to GPU
            for i in range(8):  # First 8 layers to GPU
                base_device_map[f"model.layers.{i}"] = 0
            device_map = base_device_map
        else:  # Low memory GPU or no GPU
            device_map = base_device_map
            
        # Add default mapping for any unmapped parameters
        device_map[""] = "cpu"
        
        print(f"GPU Memory Available: {gpu_memory_gb:.2f}GB")
        print(f"Using device map strategy: {'auto' if device_map == 'auto' else 'manual'}")
        
        # Initialize tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model with quantization and CPU offloading
        print("Loading base model with 4-bit quantization and CPU offloading...")
        try:
            # First attempt: load with auto device map
            if device_map == "auto":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    quantization_config=quantization_config,
                    device_map="auto",
                    max_memory={0: f"{int(gpu_memory_gb * 0.8)}GB", "cpu": "24GB"},
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                )
            else:
                # Second attempt: load with explicit device map
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    quantization_config=quantization_config,
                    device_map=device_map,
                    max_memory={0: f"{int(gpu_memory_gb * 0.8)}GB", "cpu": "24GB"},
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    offload_folder="offload",  # Enable disk offloading
                    offload_state_dict=True,   # Enable state dict offloading
                )
        except Exception as e:
            print(f"Error loading model with device map: {e}")
            print("Attempting to load model in CPU-only mode...")
            
            # Fallback: load everything on CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="cpu",
                torch_dtype=torch.float32,  # Use float32 for CPU
                low_cpu_mem_usage=True,
            )
        
        # Prepare model for k-bit training with CPU settings
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=False  # Disable for CPU training
        )
        
        # Configure LoRA for efficient fine-tuning
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,              # Reduced LoRA attention dimension for CPU
            lora_alpha=64,     # Reduced alpha for CPU
            lora_dropout=0.1,  # Dropout probability for LoRA layers
            bias="none",       # Don't train bias terms
            target_modules=[   # Target attention modules for LoRA
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj"
            ],
        )
        
        # Apply LoRA and print trainable parameters
        print("Applying LoRA adapters...")
        self.model = get_peft_model(self.model, self.lora_config)
        
        # Print parameter counts
        trainable_params = 0
        all_params = 0
        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"✨ Model prepared with QLoRA!")
        print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}% of {all_params:,} total)")

    def prepare_training_data(self, dataset_path: str = "Binary Dataset/cipher_cloud_dataset.json"):
        """Prepare dataset for fine-tuning"""
        print("Loading and preparing dataset...")
        
        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} policies from dataset")
            
            # Debug: Show the structure of the first policy
            if len(data) > 0:
                print("\nExample policy structure:")
                first_policy = data[0]
                print(json.dumps(first_policy, indent=2)[:500] + "...")
                print("\nAvailable keys in policy:", list(first_policy.keys()))
            
        except FileNotFoundError:
            print(f"❌ Dataset file not found: {dataset_path}")
            print("Checking alternative paths...")
            
            # Try alternative path
            alt_path = os.path.join("Binary Dataset", "cipher_cloud_dataset.json")
            try:
                with open(alt_path, 'r') as f:
                    data = json.load(f)
                print(f"Loaded {len(data)} policies from alternate path")
            except FileNotFoundError:
                raise FileNotFoundError(f"Could not find dataset file in {dataset_path} or {alt_path}")
        
        # Debug: Count policies with different risk values
        risk_counts = {}
        for policy in data:
            risk_value = policy.get('is_risky', 'not_specified')
            risk_counts[risk_value] = risk_counts.get(risk_value, 0) + 1
        
        print("\nRisk value distribution:")
        for risk_value, count in risk_counts.items():
            print(f"is_risky={risk_value}: {count} policies")
        
        # Separate benign and risky policies based on label (0 = benign, 1 = risky)
        benign_policies = [p for p in data if p.get('label', 1) == 0]  # Default to risky if no label
        risky_policies = [p for p in data if p.get('label', 1) == 1]
        
        print(f"\nFound {len(benign_policies)} benign policies and {len(risky_policies)} risky policies")
        
        if not benign_policies or not risky_policies:
            raise ValueError("Not enough data: need both benign and risky policies for training")
        
        # Create instruction pairs
        train_data = []
        for risky in risky_policies:
            # Find most similar benign policy based on services used
            benign = self._find_similar_benign_policy(risky, benign_policies)
            
            # Format the prompt
            prompt = (
                "Rewrite this risky IAM policy to be secure while preserving necessary permissions:\n\n"
                f"Input:\n{json.dumps(risky.get('policy', {}), indent=2)}\n\n"
                "Output:"
            )
            
            # Format the completion (target)
            completion = f"\n{json.dumps(benign.get('policy', {}), indent=2)}"
            
            train_data.append({
                "prompt": prompt,
                "completion": completion
            })
        
        # Create HF dataset
        dataset = Dataset.from_list(train_data)
        
        def tokenize_function(examples):
            print(f"Tokenizing batch of {len(examples['prompt'])} examples...")
            # Tokenize prompts and completions separately
            prompts = examples["prompt"]
            completions = examples["completion"]
            
            # Debug first example
            if len(prompts) > 0:
                print("\nExample prompt:", prompts[0][:100], "...")
                print("Example completion:", completions[0][:100], "...")
            
            # Tokenize inputs
            tokenized_inputs = self.tokenizer(
                prompts,
                truncation=True,
                max_length=1024,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Tokenize targets
            tokenized_targets = self.tokenizer(
                completions,
                truncation=True,
                max_length=1024,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Prepare input_ids and labels for causal language modeling
            input_ids = []
            labels = []
            attention_mask = []
            
            for i in range(len(prompts)):
                # Combine prompt and completion tokens
                combined_input_ids = torch.cat([
                    tokenized_inputs["input_ids"][i],
                    tokenized_targets["input_ids"][i]
                ])
                
                # Create labels (-100 for prompt tokens, actual ids for completion tokens)
                combined_labels = torch.cat([
                    torch.full_like(tokenized_inputs["input_ids"][i], -100),
                    tokenized_targets["input_ids"][i]
                ])
                
                # Combine attention masks
                combined_attention_mask = torch.cat([
                    tokenized_inputs["attention_mask"][i],
                    tokenized_targets["attention_mask"][i]
                ])
                
                input_ids.append(combined_input_ids)
                labels.append(combined_labels)
                attention_mask.append(combined_attention_mask)
            
            return {
                "input_ids": torch.stack(input_ids),
                "attention_mask": torch.stack(attention_mask),
                "labels": torch.stack(labels)
            }
        
        # Process dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            load_from_cache_file=False
        )
        
        return tokenized_dataset

    def _find_similar_benign_policy(self, risky_policy: Dict, benign_policies: List[Dict]) -> Dict:
        """Find most similar benign policy based on AWS services used"""
        risky_services = self._extract_services(risky_policy.get('policy', {}))
        
        best_match = None
        best_score = -1
        
        for benign in benign_policies:
            benign_services = self._extract_services(benign.get('policy', {}))
            
            # Calculate Jaccard similarity
            similarity = len(risky_services & benign_services) / len(risky_services | benign_services) if risky_services or benign_services else 0
            
            if similarity > best_score:
                best_score = similarity
                best_match = benign
        
        return best_match or benign_policies[0]

    def _extract_services(self, policy: Dict) -> set:
        """Extract AWS service names from policy actions"""
        services = set()
        statements = policy.get('Statement', [])
        if not isinstance(statements, list):
            statements = [statements]
            
        for stmt in statements:
            actions = stmt.get('Action', [])
            if isinstance(actions, str):
                actions = [actions]
            
            for action in actions:
                if ':' in action:
                    service = action.split(':')[0]
                    services.add(service)
        
        return services

    def train(self, dataset, output_dir: str = "Models/policy_llm_finetuned"):
        """Fine-tune the model on our policy dataset using QLoRA"""
        
        # Determine optimal batch size and gradient accumulation based on available memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        gpu_memory_gb = gpu_memory / (1024 ** 3)
        
        if gpu_memory_gb >= 24:  # High-end GPU
            batch_size = 4
            gradient_accumulation = 8
        elif gpu_memory_gb >= 8:  # Mid-range GPU
            batch_size = 1
            gradient_accumulation = 32
        else:  # Low memory GPU or no GPU
            batch_size = 1
            gradient_accumulation = 64
        
        print(f"\nTraining Configuration:")
        print(f"Batch size: {batch_size}")
        print(f"Gradient accumulation steps: {gradient_accumulation}")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,  # Small batch size for CPU
            gradient_accumulation_steps=16,  # Reduced for CPU
            learning_rate=5e-5,             # Reduced learning rate for CPU
            fp16=False,                     # Disable fp16 for CPU
            logging_steps=1,
            save_strategy="epoch",
            warmup_steps=50,
            lr_scheduler_type="linear",      # Simpler scheduler for CPU
            report_to="none",
            gradient_checkpointing=False,    # Disable for CPU
            optim="adamw_torch",            # Use standard optimizer for CPU
            max_grad_norm=0.3,
            remove_unused_columns=False,
            group_by_length=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,        # Single worker for CPU
            do_eval=False,                   # No evaluation during training
            save_total_limit=1,             # Keep only the latest checkpoint
            use_cpu=True,                   # Force CPU usage
            bf16=False,                     # Disable mixed precision
            ddp_find_unused_parameters=None,
            ddp_bucket_cap_mb=None,
            deepspeed=None,
            local_rank=-1                   # Single CPU training
        )

        # Custom data collator that handles both tensor and list inputs
        def data_collator(features):
            # Convert lists to tensors if necessary
            def to_tensor(x):
                if isinstance(x, torch.Tensor):
                    return x
                return torch.tensor(x)
            
            return {
                'input_ids': torch.stack([to_tensor(f['input_ids']) for f in features]),
                'attention_mask': torch.stack([to_tensor(f['attention_mask']) for f in features]),
                'labels': torch.stack([to_tensor(f['labels']) for f in features])
            }

        # Sanity check dataset
        print(f"\nDataset statistics:")
        print(f"Number of examples: {len(dataset)}")
        if len(dataset) == 0:
            raise ValueError("Dataset is empty! Check the data preparation pipeline.")
            
        # Check first example and dataset setup
        first_example = dataset[0]
        print("\nFirst example contents:")
        for key, value in first_example.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
            elif isinstance(value, list):
                print(f"{key}: list of length {len(value)}")
            else:
                print(f"{key}: {type(value)}")

        # Validate all examples have the required keys
        required_keys = {'input_ids', 'attention_mask', 'labels'}
        for i, example in enumerate(dataset):
            missing_keys = required_keys - set(example.keys())
            if missing_keys:
                print(f"\nWARNING: Example {i} is missing required keys: {missing_keys}")
                break

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )

        print("Starting training...")
        trainer.train()
        
        # Save the fine-tuned model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

def main():
    # Check workspace structure
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(workspace_dir, "Binary Dataset", "cipher_cloud_dataset.json")
    output_dir = os.path.join(workspace_dir, "Models", "policy_llm_finetuned")
    
    print(f"\nWorkspace directory: {workspace_dir}")
    print(f"Looking for dataset at: {dataset_path}")
    print(f"Model will be saved to: {output_dir}")
    
    # Initialize trainer
    trainer = PolicyLLMTrainer()
    
    # Prepare dataset
    dataset = trainer.prepare_training_data(dataset_path=dataset_path)
    
    # Train model
    trainer.train(dataset, output_dir=output_dir)
    
    print("✨ Fine-tuning complete! Saving LoRA weights and merged model...")
    
    # Save LoRA weights
    lora_path = os.path.join(output_dir, "lora_weights")
    trainer.model.save_pretrained(lora_path)
    
    # Save tokenizer
    trainer.tokenizer.save_pretrained(lora_path)
    
    # Merge weights and save full model (optional, memory intensive)
    try:
        print("Attempting to merge and save full model...")
        merged_path = os.path.join(output_dir, "merged_model")
        
        # Merge LoRA weights with base model
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(merged_path)
        trainer.tokenizer.save_pretrained(merged_path)
        print(f"Full merged model saved to: {merged_path}")
    except Exception as e:
        print(f"Note: Could not save merged model (this is normal if memory constrained): {e}")
        print(f"LoRA weights saved separately to: {lora_path}")
    
    print("\nTo use the fine-tuned model, update policy_rewrite.py with either:")
    print(f"1. LoRA weights path: {lora_path}")
    print(f"2. Full merged model path: {merged_path} (if available)")
    print("\nDone! The model has been trained on secure vs risky policies. 🎉")

if __name__ == "__main__":
    main()
