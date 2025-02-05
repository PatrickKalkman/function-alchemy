import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any

# Example OpenAI-style function definitions
K8S_FUNCTIONS = [
    {
        "name": "get_number_of_nodes",
        "description": "Get the number of nodes in the Kubernetes cluster",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_last_events",
        "description": "What's been happening in the cluster lately?",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_version_info",
        "description": "Returns version information for both Kubernetes API server and nodes.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]

PROMPT_TEMPLATE = """You are an AI that ONLY returns the selected function call in JSON format. No explanations, no reasoning, only JSON.

Available Functions:
{functions}

Instruction: {instruction}"""


def load_base_model(model_name: str = "microsoft/phi-2"):
    """Load the base Phi-2 model without fine-tuning."""
    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def format_prompt(instruction: str, available_functions: List[Dict[str, Any]]) -> str:
    """Format the prompt with OpenAI-style function definitions."""
    functions_str = json.dumps(available_functions, indent=2)
    return PROMPT_TEMPLATE.format(functions=functions_str, instruction=instruction)


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512):
    """Generate a response from the base model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print("\nGenerating response:")
    print("-" * 40)

    output_tokens = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    ).sequences

    response_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print(response_text)
    print("-" * 40)
    return response_text.replace(prompt, "").strip()


def run_base_model_tests():
    """Run test cases using the base Phi-2 model."""
    model, tokenizer = load_base_model()

    test_cases = [
        "How many nodes are in the cluster?",
        "What's been happening in the cluster lately?",
        "Are my nodes all on the same version?",
        "Check cluster version",
    ]

    for instruction in test_cases:
        print(f"\nTesting: {instruction}")
        prompt = format_prompt(instruction, K8S_FUNCTIONS)
        response = generate_response(model, tokenizer, prompt)
        print(f"Raw response:\n{response}")


if __name__ == "__main__":
    run_base_model_tests()
