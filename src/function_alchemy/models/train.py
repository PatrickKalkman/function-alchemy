import os
import torch
from huggingface_hub import login
import wandb
from trl import SFTTrainer
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import pandas as pd
from datasets import Dataset

from dotenv import load_dotenv

load_dotenv()

# --- 1. Setup and Authentication ---
# Read API keys from environment variables
hf_token = os.environ.get("HUGGINGFACE_TOKEN")
wb_token = os.environ.get("WANDB_API_KEY")

if not hf_token:
    raise ValueError("HUGGINGFACE_TOKEN environment variable not set.")
if not wb_token:
    raise ValueError("WANDB_API_KEY environment variable not set.")

login(token=hf_token)
wandb.login(key=wb_token)

# Initialize Weights & Biases (wandb)
run = wandb.init(project="DeepSeek-R1-Distill-Qwen-1.5B-func", job_type="training", anonymous="allow")

# --- 2. Model and Tokenizer Loading ---
max_seq_length = 2048
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # set to appropriate dtype. set to torch.float16 if bfloat16 is not supported
    # load_in_4bit=True
)
# --- 3. Function Calling Specific Prompt and Dataset ---
# Define a prompt template that includes placeholders for function call data
train_prompt_style = """Below is an instruction that describes a task, paired with
 an input that provides further context.
 Write a response that appropriately completes the request.
 Before answering, think carefully about the question and create a step-by-step chain
 of thoughts to ensure a logical and accurate response.
If the request requires calling a function, write out the function call
 in the function call format {{"name":"function_name", "arguments": {{ "arg1": "value1", "arg2": "value2" }}}}.

### Instruction:
You are a helpful assistant capable of completing tasks including function calls.

### Question:
{}

### Response:
<think>
{}
</think>
{}"""

EOS_TOKEN = tokenizer.eos_token  # Use EOS token from tokenizer


def format_function_calling_data(examples):
    inputs = examples["instruction"]
    cots = examples["thought"]
    outputs = examples["output"]  # This could be regular text or a formatted function call
    texts = []
    for input, cot, output in zip(inputs, cots, outputs):
        text = train_prompt_style.format(input, cot, output) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }


# Example Function Call Data
function_calling_data = [
    {
        "instruction": "What's the weather in London?",
        "thought": "I need to use the weather tool to get the forecast.",
        "output": '{"name":"get_weather", "arguments": {"location": "London"}}',
    },
    {
        "instruction": "What's the current time in Tokyo",
        "thought": "I need to use the time tool to get the current time.",
        "output": '{"name":"get_time", "arguments": {"location": "Tokyo"}}',
    },
    {
        "instruction": "What is 23 + 12?",
        "thought": "I can do this math without needing a tool",
        "output": "23 + 12 equals 35.",
    },
]

dataset = pd.DataFrame(function_calling_data)
dataset = dataset.to_dict("records")


dataset = Dataset.from_list(dataset)

# Apply the prompt formatting
dataset = dataset.map(format_function_calling_data, batched=True)

print(dataset["text"][0])

# --- 4. Preparing the Model for Fine-tuning ---
# Get the Lora PEFT model
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

model = get_peft_model(model, lora_config)

# --- 5. Trainer Setup and Training ---
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    data_collator=data_collator,  # added data_collator here
    args=TrainingArguments(
        run_name="DeepSeek-R1-Distill-Qwen-1.5B-func-12",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=False,  # Use fp16 instead of bf16 or remove if GPU does not support fp16
        logging_steps=10,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

# Train the model
trainer_stats = trainer.train()

# --- 6. Model Saving and Pushing ---
wandb.finish()

# Define model save paths
new_model_online = "pkalkman/DeepSeek-R1-Distill-Qwen-1.5B-func"
new_model_local = "DeepSeek-R1-Distill-Qwen-1.5B-func"

# Save the trained LoRA adapters locally
model.save_pretrained(new_model_local)
tokenizer.save_pretrained(new_model_local)

# Push the LoRA adapters to the Hugging Face Hub
model.push_to_hub(new_model_online)
tokenizer.push_to_hub(new_model_online)

print("Training Done")
