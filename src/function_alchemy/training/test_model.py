import torch
import json
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional
from unsloth import FastLanguageModel

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

PROMPT_TEMPLATE = """Return exactly one function call in JSON format with name and parameters fields. Example:
{{"name": "function_name", "parameters": {{"type": "object", "properties": {{}}, "required": []}}}}

Available Functions to choose from:
{functions}

Function: {instruction}"""


def load_model(model_repo: str, base_model_name: str = "microsoft/phi-4"):
    """Load the fine-tuned model and tokenizer using unsloth optimizations."""
    try:
        print(f"Loading model and tokenizer using unsloth from: {model_repo}")

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Load the base model with unsloth optimizations
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=256,
            dtype="bfloat16",
            load_in_4bit=True,
            cache_dir="./cache",
        )

        # Load the trained LoRA weights
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            peft_model_id=model_repo,
        )

        model.eval()
        return model, tokenizer

    except Exception as e:
        print(f"Failed to load model: {e}")
        raise


def format_prompt(instruction: str, available_functions: List[Dict[str, Any]]) -> str:
    """Format the prompt with OpenAI-style function definitions."""
    functions_str = json.dumps(available_functions, indent=2)
    return PROMPT_TEMPLATE.format(functions=functions_str, instruction=instruction)


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 128):
    """Generate a response from the model with streaming output."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generated_text = ""
    print("\nGenerating response:")
    print("-" * 40)

    generation_output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )

    for output in generation_output.sequences:
        current_text = tokenizer.decode(output, skip_special_tokens=True)
        new_text = current_text[len(generated_text) :]
        print(new_text, end="", flush=True)
        generated_text = current_text

    print("\n" + "-" * 40)
    return generated_text.replace(prompt, "").strip()


def parse_function_call(response: str) -> Optional[Dict[str, Any]]:
    """Extract and parse the function call from the model's response."""
    try:
        response = response.replace("Output: ", "").strip()
        parsed = json.loads(response)

        if "name" in parsed:
            function_name = parsed["name"]
            arguments = parsed.get("arguments", {})
            transformed = {"function_call": {"name": function_name, "arguments": arguments}}
            return transformed

        return parsed
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {response}")
        return None


def run_test_cases():
    """Run test cases with flexible function calling format validation."""
    model_path = "pkalkman/phi-4-func"  # Updated to your new model name
    model, tokenizer = load_model(model_path)

    test_cases = [
        {
            "instruction": "How many nodes are in the cluster?",
            "expected_function": "get_number_of_nodes",
            "schema": {
                "name": "get_number_of_nodes",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "instruction": "What's been happening in the cluster lately?",
            "expected_function": "get_last_events",
            "schema": {
                "name": "get_last_events",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "instruction": "Are my nodes all on the same version?",
            "expected_function": "get_version_info",
            "schema": {
                "name": "get_version_info",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "instruction": "Check cluster version",
            "expected_function": "get_version_info",
            "schema": {
                "name": "get_version_info",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]

    results = []
    for test_case in test_cases:
        print(f"\nTesting: {test_case['instruction']}")
        prompt = format_prompt(test_case["instruction"], K8S_FUNCTIONS)
        response = generate_response(model, tokenizer, prompt)
        print(f"Raw response:\n{response}")

        function_call = parse_function_call(response)
        print(f"Parsed function call:\n{json.dumps(function_call, indent=2) if function_call else None}")

        actual_name = None
        if function_call and "function_call" in function_call:
            actual_name = function_call["function_call"].get("name")
        elif function_call and "name" in function_call:
            actual_name = function_call["name"]

        success = actual_name == test_case["expected_function"]

        results.append(
            {
                "instruction": test_case["instruction"],
                "expected_function": test_case["expected_function"],
                "actual_function": actual_name,
                "success": success,
                "error": None if success else f"Expected {test_case['expected_function']}, got {actual_name}",
            }
        )

        status = "✓" if success else "✗"
        print(f"{status} Function name match: {success}")

    print("\nTest Summary:")
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    print(f"Passed: {passed}/{total} ({passed / total * 100:.1f}%)")

    return results


if __name__ == "__main__":
    results = run_test_cases()
