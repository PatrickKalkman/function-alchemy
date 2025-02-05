"""
Main training script for fine-tuning the model on Kubernetes function calling using OpenAI spec.
"""

import os
import torch
import json
from huggingface_hub import login
import wandb
from trl import SFTTrainer
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from dotenv import load_dotenv

from ..data.loader import load_training_data, PROMPT_TEMPLATE

load_dotenv()


def setup_wandb():
    """Setup Weights & Biases tracking."""
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    wb_token = os.environ.get("WANDB_API_KEY")

    if not hf_token or not wb_token:
        raise ValueError("Missing required environment variables")

    login(token=hf_token)
    wandb.login(key=wb_token)

    return wandb.init(project="DeepSeek-R1-Distill-Qwen-1.5B-func", job_type="training", anonymous="allow")


def load_model_and_tokenizer(model_name):
    """Load the base model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"

    # Add device map for Mac M3
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Changed from bfloat16 to float16 for M3
        device_map="auto",  # Enable automatic device mapping
        use_cache=False,  # Reduce memory usage during training
    )
    return model, tokenizer


def format_function_calling_data(examples, tokenizer):
    """Format the training data according to our prompt template."""
    texts = []
    for input, cot, output, funcs in zip(
        examples["instruction"], examples["thought"], examples["output"], examples["functions"]
    ):
        # Convert to pretty-printed strings
        functions_str = json.dumps(funcs, indent=2)
        output_str = json.dumps(output, indent=2)

        text = (
            PROMPT_TEMPLATE.format(functions=functions_str, instruction=input, thought=cot, output=output_str)
            + tokenizer.eos_token
        )

        texts.append(text)
    return {"text": texts}


def prepare_datasets(data, tokenizer):
    """Prepare and split the datasets."""
    # Convert to Dataset format
    dataset = Dataset.from_list(data)

    # Split into train and evaluation sets
    splits = dataset.train_test_split(test_size=0.2, shuffle=True, seed=3407)

    # Format both splits
    train_dataset = splits["train"].map(
        lambda x: format_function_calling_data(x, tokenizer), batched=True, remove_columns=splits["train"].column_names
    )

    eval_dataset = splits["test"].map(
        lambda x: format_function_calling_data(x, tokenizer), batched=True, remove_columns=splits["test"].column_names
    )

    print("\nDataset sizes:")
    print(f"Training examples: {len(train_dataset)}")
    print(f"Evaluation examples: {len(eval_dataset)}")

    return train_dataset, eval_dataset


def get_training_args():
    return TrainingArguments(
        run_name="DeepSeek-R1-Distill-Qwen-1.5B-func-openai",
        per_device_train_batch_size=1,  # Reduced for M3
        per_device_eval_batch_size=1,  # Reduced for M3
        gradient_accumulation_steps=16,  # Increased to compensate
        max_steps=100,  # Reduced for testing
        learning_rate=1e-4,
        fp16=False,  # Enable mixed precision
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        output_dir="outputs",
        save_total_limit=2,  # Limit saved checkpoints
        push_to_hub=False,  # Disable during testing
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
    )


def prepare_model_for_training(model):
    """Optimize model for training on M3"""
    model.config.use_cache = False

    # Enable gradient checkpointing
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return model


if __name__ == "__main__":
    # Load model and tokenizer
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model, tokenizer = load_model_and_tokenizer(model_name)
    model = prepare_model_for_training(model)

    # Get the training data
    function_calling_data = load_training_data()
    train_dataset, eval_dataset = prepare_datasets(function_calling_data, tokenizer)

    # Setup LoRA configuration
    lora_config = LoraConfig(
        r=8,  # Reduced rank
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA config to model
    model = get_peft_model(model, lora_config)

    # Initialize wandb
    run = setup_wandb()

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        args=get_training_args(),
    )

    # Train the model
    trainer.train()

    # Save the model
    new_model_name = "DeepSeek-R1-Distill-Qwen-1.5B-func-openai"
    model.save_pretrained(new_model_name)
    tokenizer.save_pretrained(new_model_name)

    # Push to Hub if desired
    if os.environ.get("PUSH_TO_HUB", "true").lower() == "true":
        model.push_to_hub(f"pkalkman/{new_model_name}")
        tokenizer.push_to_hub(f"pkalkman/{new_model_name}")

    # Cleanup
    wandb.finish()
    print("\nTraining completed successfully!")
