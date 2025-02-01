import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your model and tokenizer from local directory or the hub
model_path = "DeepSeek-R1-Distill-Qwen-1.5B-func"  # or your hub model name
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# If you're using an MPS device on macOS:
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

# Define a prompt (this can mimic your training prompt if needed)
prompt = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step
 chain of thoughts to ensure a logical and accurate response.
If the request requires calling a function, write out the function call in the
 function call format {"name":"function_name", "arguments": {"arg1": "value1", "arg2": "value2"}}.

### Instruction:
You are a helpful assistant capable of completing tasks including function calls.

### Question:
What's the weather in London?

### Response:
<think>
"""

# Tokenize and generate a response
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(result)
