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

# Import our custom modules
from function_alchemy.data.k8s_functions_config import PROMPT_TEMPLATE
from function_alchemy.data.k8s_data_generator import get_combined_training_data

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

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )

    return model, tokenizer


def format_function_calling_data(examples):
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


def prepare_datasets(data):
    """Prepare and split the datasets."""
    # Convert to Dataset format
    dataset = Dataset.from_list(data)

    # Split into train and evaluation sets
    splits = dataset.train_test_split(test_size=0.2, shuffle=True, seed=3407)

    # Format both splits
    train_dataset = splits["train"].map(
        format_function_calling_data, batched=True, remove_columns=splits["train"].column_names
    )

    eval_dataset = splits["test"].map(
        format_function_calling_data, batched=True, remove_columns=splits["test"].column_names
    )

    print("\nDataset sizes:")
    print(f"Training examples: {len(train_dataset)}")
    print(f"Evaluation examples: {len(eval_dataset)}")

    # Print samples
    print("\nSample training example:")
    print(train_dataset[0]["text"][:500] + "...")
    print("\nSample evaluation example:")
    print(eval_dataset[0]["text"][:500] + "...")

    return train_dataset, eval_dataset


def get_training_args():
    """Get training arguments."""
    return TrainingArguments(
        run_name="DeepSeek-R1-Distill-Qwen-1.5B-func-openai",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        max_steps=1000,
        learning_rate=1e-4,
        fp16=False,
        logging_steps=50,
        save_steps=200,
        eval_steps=200,
        eval_strategy="steps",
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )


if __name__ == "__main__":
    # Initialize wandb
    run = setup_wandb()

    # Load model and tokenizer
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Get the training data
    function_calling_data = get_combined_training_data()

    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(function_calling_data)

    # Setup LoRA configuration
    lora_config = LoraConfig(
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA config to model
    model = get_peft_model(model, lora_config)

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
