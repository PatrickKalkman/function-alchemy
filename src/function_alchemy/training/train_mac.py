import os
import torch
import json
from huggingface_hub import login
import wandb
from transformers import TrainingArguments, AutoTokenizer, DataCollatorForLanguageModeling, Trainer
from datasets import Dataset
from dotenv import load_dotenv
import huggingface_hub
from unsloth import FastLanguageModel
from peft import LoraConfig, prepare_model_for_kbit_training
from ..data.loader import load_training_data, PROMPT_TEMPLATE

huggingface_hub.constants.HUGGINGFACE_HUB_DEFAULT_TIMEOUT = 60
load_dotenv()


def setup_wandb():
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    wb_token = os.environ.get("WANDB_API_KEY")
    if not hf_token or not wb_token:
        raise ValueError("Missing required environment variables")
    login(token=hf_token)
    wandb.login(key=wb_token)
    return wandb.init(project="phi-4-func", job_type="training", anonymous="allow")


def load_model_and_tokenizer(model_name):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # First load base model with unsloth optimizations
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=256,
        dtype="bfloat16",
        load_in_4bit=True,
        cache_dir="./cache",
    )

    # Then apply LoRA config using unsloth's FastLora
    from unsloth.fast_lora import FastLora

    lora = FastLora(
        model,
        r=8,
        alpha=16,
        dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = lora.merge_and_unload()
    model.print_trainable_parameters()

    return model, tokenizer


def format_function_calling_data(examples, tokenizer):
    texts = []
    for input, output, funcs in zip(examples["instruction"], examples["output"], examples["functions"]):
        try:
            output_str = json.dumps(output, indent=2)
            functions_str = json.dumps(funcs, indent=2)
            text = (
                PROMPT_TEMPLATE.format(functions=functions_str, instruction=input, output=output_str)
                + tokenizer.eos_token
            )
            texts.append(text)
        except (TypeError, json.JSONDecodeError):
            continue
    return {"text": texts}


def prepare_datasets(data, tokenizer):
    dataset = Dataset.from_list(data)
    splits = dataset.train_test_split(test_size=0.2, shuffle=True, seed=3407)

    train_dataset = splits["train"].map(
        lambda x: format_function_calling_data(x, tokenizer), batched=True, remove_columns=splits["train"].column_names
    )

    eval_dataset = splits["test"].map(
        lambda x: format_function_calling_data(x, tokenizer), batched=True, remove_columns=splits["test"].column_names
    )

    print(f"\nTraining examples: {len(train_dataset)}")
    print(f"Evaluation examples: {len(eval_dataset)}")
    return train_dataset, eval_dataset


def get_training_args():
    return TrainingArguments(
        run_name="phi-4-func",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        max_steps=500,
        learning_rate=2e-5,
        weight_decay=0.1,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.2,
        fp16=True,
        logging_steps=10,
        eval_strategy="steps",  # Updated from evaluation_strategy
        eval_steps=10,
        save_strategy="steps",
        save_steps=50,
        output_dir="outputs",
        save_total_limit=2,
        push_to_hub=False,
        report_to=["wandb"],
        gradient_checkpointing=True,
    )


if __name__ == "__main__":
    model_name = "microsoft/phi-4"
    model, tokenizer = load_model_and_tokenizer(model_name)

    function_calling_data = load_training_data()
    train_dataset, eval_dataset = prepare_datasets(function_calling_data, tokenizer)

    # Initialize wandb
    run = setup_wandb()

    # Get training arguments
    training_args = get_training_args()

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Start training
    trainer.train()

    # Save the model
    new_model_name = "phi-4-func"
    trainer.save_model(new_model_name)
    tokenizer.save_pretrained(new_model_name)

    if os.environ.get("PUSH_TO_HUB", "true").lower() == "true":
        model.push_to_hub(f"pkalkman/{new_model_name}")
        tokenizer.push_to_hub(f"pkalkman/{new_model_name}")

    wandb.finish()
    print("\nTraining completed successfully!")
