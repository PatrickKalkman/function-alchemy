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
import huggingface_hub
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
    return wandb.init(project="phi-2-2.7B-func", job_type="training", anonymous="allow")


def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=False,
        max_length=256,
        do_sample=False,
        temperature=0.1,
    )
    return model, tokenizer


def format_function_calling_data(examples, tokenizer):
    texts = []
    for input, output, funcs in zip(examples["instruction"], examples["output"], examples["functions"]):
        # Strict JSON validation
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
        run_name="phi-2-func",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        max_steps=100,
        learning_rate=5e-5,
        weight_decay=0.05,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=False,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=50,
        output_dir="outputs",
        save_total_limit=2,
        push_to_hub=False,
        report_to=["wandb"],
    )


def prepare_model_for_training(model):
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


if __name__ == "__main__":
    model_name = "microsoft/phi-2"
    model, tokenizer = load_model_and_tokenizer(model_name)
    model = prepare_model_for_training(model)

    function_calling_data = load_training_data()
    train_dataset, eval_dataset = prepare_datasets(function_calling_data, tokenizer)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    run = setup_wandb()

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        args=get_training_args(),
    )

    trainer.train()

    new_model_name = "phi-2-2.7B-func"
    model.save_pretrained(new_model_name)
    tokenizer.save_pretrained(new_model_name)

    if os.environ.get("PUSH_TO_HUB", "true").lower() == "true":
        model.push_to_hub(f"pkalkman/{new_model_name}")
        tokenizer.push_to_hub(f"pkalkman/{new_model_name}")

    wandb.finish()
    print("\nTraining completed successfully!")
