import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import List, Dict, Any, Optional

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

PROMPT_TEMPLATE = """You are an AI that ONLY returns the selected single function call in JSON format. No explanations, no reasoning, only JSON.

Available Functions:
{functions}

Instruction: {instruction}"""


def load_model(model_repo: str, base_model_name: str = "microsoft/phi-2"):
    """Load the fine-tuned model and tokenizer, first trying locally then from HuggingFace Hub."""
    try:
        # Try loading base model from HuggingFace Hub first
        print(f"Loading base model from HuggingFace Hub: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"HuggingFace Hub load failed: {e}")
        print(f"Attempting to load base model locally from {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, local_files_only=True)

    try:
        # Try loading PEFT adapter from HuggingFace Hub
        print(f"Loading PEFT adapter from HuggingFace Hub: {model_repo}")
        model = PeftModel.from_pretrained(model, model_repo, device_map="auto")
    except Exception as e:
        print(f"HuggingFace Hub PEFT load failed: {e}")
        print(f"Attempting to load PEFT adapter locally from {model_repo}")
        model = PeftModel.from_pretrained(model, model_repo, device_map="auto", local_files_only=True)

    model.eval()
    return model, tokenizer


def format_prompt(instruction: str, available_functions: List[Dict[str, Any]]) -> str:
    """Format the prompt with OpenAI-style function definitions."""
    functions_str = json.dumps(available_functions, indent=2)
    return PROMPT_TEMPLATE.format(functions=functions_str, instruction=instruction)


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512):
    """Generate a response from the model with streaming output."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # print("\nPrompt:")
    # print(prompt)
    # Store the generated text
    generated_text = ""
    print("\nGenerating response:")
    print("-" * 40)

    # Stream the output token by token
    for output in model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=None,  # We'll handle streaming manually
        return_dict_in_generate=True,
        output_scores=True,
    ).sequences:
        current_text = tokenizer.decode(output, skip_special_tokens=True)
        new_text = current_text[len(generated_text) :]
        print(new_text, end="", flush=True)
        generated_text = current_text

    print("\n" + "-" * 40)

    # Return only the generated response without the prompt
    return generated_text.replace(prompt, "").strip()


def parse_function_call(response: str) -> Optional[Dict[str, Any]]:
    """Extract and parse the function call from the model's response."""
    try:
        response = response.replace("Output: ", "").strip()
        # Parse the JSON
        parsed = json.loads(response)

        # Transform the format if needed
        if "name" in parsed:
            # Convert from model's format to OpenAI format
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
    model_path = "pkalkman/phi-2-2.7B-func"
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

        # Format prompt with all available functions
        prompt = format_prompt(test_case["instruction"], K8S_FUNCTIONS)

        # Generate response
        response = generate_response(model, tokenizer, prompt)
        print(f"Raw response:\n{response}")

        # Parse function call
        function_call = parse_function_call(response)
        print(f"Parsed function call:\n{json.dumps(function_call, indent=2) if function_call else None}")

        # Get the actual function name from the parsed response
        actual_name = None
        if function_call and "function_call" in function_call:
            actual_name = function_call["function_call"].get("name")
        elif function_call and "name" in function_call:
            actual_name = function_call["name"]

        # Validate function name
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

        # Print individual test result
        status = "✓" if success else "✗"
        print(f"{status} Function name match: {success}")

    # Print summary
    print("\nTest Summary:")
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    print(f"Passed: {passed}/{total} ({passed / total * 100:.1f}%)")

    return results


if __name__ == "__main__":
    results = run_test_cases()
