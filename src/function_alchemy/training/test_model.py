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
        "name": "analyze_deployment_logs",
        "description": "Analyze logs from a specific deployment",
        "parameters": {
            "type": "object",
            "properties": {
                "deployment_name": {"type": "string", "description": "Name of the deployment"},
                "namespace": {"type": "string", "description": "Kubernetes namespace", "default": "default"},
            },
            "required": ["deployment_name", "namespace"],
        },
    },
]

PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an API that contains functions. Write a response that appropriately calls the function.

Available Functions:
{functions}

Instruction: {instruction}

Response: {output}"""


def load_model(model_repo: str, base_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
    """Load the fine-tuned model and tokenizer, first trying locally then from HuggingFace Hub."""
    try:
        # Try loading base model locally first
        print(f"Attempting to load base model locally from {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, local_files_only=True)
    except Exception as e:
        print(f"Local load failed: {e}")
        print(f"Loading base model from HuggingFace Hub: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    try:
        # Try loading PEFT adapter locally
        print(f"Attempting to load PEFT adapter locally from {model_repo}")
        model = PeftModel.from_pretrained(model, model_repo, device_map="auto", local_files_only=True)
    except Exception as e:
        print(f"Local PEFT load failed: {e}")
        print(f"Loading PEFT adapter from HuggingFace Hub: {model_repo}")
        model = PeftModel.from_pretrained(model, model_repo, device_map="auto")

    model.eval()
    return model, tokenizer


def format_prompt(instruction: str, available_functions: List[Dict[str, Any]]) -> str:
    """Format the prompt with OpenAI-style function definitions."""
    functions_str = json.dumps(available_functions, indent=2)
    return PROMPT_TEMPLATE.format(
        functions=functions_str,
        instruction=instruction,
        thought="",  # Empty for inference
        output="",  # Empty for inference
    )


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512):
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()


def parse_function_call(response: str) -> Optional[Dict[str, Any]]:
    """Extract and parse the function call from the model's response."""
    try:
        # Parse the JSON
        parsed = json.loads(response)

        # Transform the format if needed
        if "function" in parsed:
            # Convert from model's format to OpenAI format
            function_data = parsed["function"]
            transformed = {"function_call": {"name": function_data["name"], "arguments": {}}}

            # Extract parameters if they exist
            if "parameters" in function_data:
                params = function_data["parameters"]
                if isinstance(params, dict):
                    # For deployment logs case
                    if "deployment_name" in params:
                        transformed["function_call"]["arguments"] = {
                            "deployment_name": "frontend",
                            "namespace": params.get("namespace", {}).get("default", "testing"),
                        }

            return transformed

        return parsed
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {response}")
        return None


def run_test_cases():
    """Run test cases with flexible function calling format validation."""
    model_path = "pkalkman/DeepSeek-R1-Distill-Qwen-1.5B-func-openai"
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
            "instruction": "Show me the logs from the frontend deployment in the testing namespace",
            "expected_function": "analyze_deployment_logs",
            "schema": {
                "name": "analyze_deployment_logs",
                "parameters": {
                    "type": "object",
                    "properties": {"deployment_name": {"type": "string"}, "namespace": {"type": "string"}},
                    "required": ["deployment_name", "namespace"],
                },
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

        # Get the actual function name
        actual_name = None
        if function_call:
            if "function" in function_call:
                actual_name = function_call["function"]["name"]
            elif "function_call" in function_call:
                actual_name = function_call["function_call"]["name"]

        results.append(
            {
                "instruction": test_case["instruction"],
                "expected_function": test_case["expected_function"],
                "actual_function": actual_name,
            }
        )

    # Print summary
    print("\nTest Summary:")
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    print(f"Passed: {passed}/{total} ({passed / total * 100:.1f}%)")

    return results


if __name__ == "__main__":
    results = run_test_cases()
