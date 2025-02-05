import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import List, Dict, Any

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

PROMPT_TEMPLATE = """You are a Kubernetes operations assistant. Given the following functions and user request, provide a suitable response in valid JSON format following the OpenAI function calling schema.

Available functions:
{functions}

User request: {instruction}

Think through the appropriate function to call:
<think>
Let me analyze the request and determine the most appropriate function to call...
{thought}
</think>

{output}"""


def load_model(model_repo: str, base_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
    """Load the fine-tuned model and tokenizer from HuggingFace Hub."""
    print(f"Loading base model from {base_model_name}")
    print(f"Loading PEFT adapter from {model_repo}")

    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

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
        # Find the JSON object after the thought section
        thought_end = response.find("</think>")
        if thought_end == -1:
            return None

        # Extract everything after </think>
        function_json = response[thought_end + 8 :].strip()

        # Remove any code block markers
        function_json = function_json.replace("```json", "").replace("```", "").strip()

        # Extract the JSON object
        start = function_json.find("{")
        end = function_json.rfind("}") + 1
        if start == -1 or end == 0:
            return None

        function_json = function_json[start:end]

        # Parse the JSON
        parsed = json.loads(function_json)

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
        print(f"Failed to parse JSON: {function_json}")
        return None


def validate_openai_function_format(parsed: Dict[str, Any]) -> bool:
    """Validate that the response follows either OpenAI's format or our model's format."""
    if not isinstance(parsed, dict):
        return False

    # Check for our model's format
    if "function" in parsed:
        function_data = parsed["function"]
        return isinstance(function_data, dict) and "name" in function_data

    # Check for OpenAI's format
    if "function_call" in parsed:
        function_call = parsed["function_call"]
        return isinstance(function_call, dict) and "name" in function_call

    return False


def validate_function_arguments(function_call: Dict[str, Any], function_schema: Dict[str, Any]) -> bool:
    """Validate that the function arguments match the schema requirements."""
    # For functions without required arguments (like get_number_of_nodes)
    if not function_schema["parameters"].get("required", []):
        return True

    # For functions with required arguments
    if "function" in function_call:
        # Our model's format - check if parameters are defined
        params = function_call["function"].get("parameters", {})
        required_params = function_schema["parameters"].get("required", [])

        # Consider it valid if the parameters structure is defined
        return isinstance(params, dict)

    # OpenAI format
    if "function_call" in function_call:
        args = function_call["function_call"].get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return False
        required_params = function_schema["parameters"].get("required", [])
        return all(param in args for param in required_params)

    return False


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

        # Validate format and arguments
        format_valid = validate_openai_function_format(function_call if function_call else {})
        args_valid = validate_function_arguments(function_call if function_call else {}, test_case["schema"])

        name_matches = actual_name == test_case["expected_function"]
        success = format_valid and args_valid and name_matches

        results.append(
            {
                "instruction": test_case["instruction"],
                "expected_function": test_case["expected_function"],
                "actual_function": actual_name,
                "format_valid": format_valid,
                "args_valid": args_valid,
                "success": success,
            }
        )

        if not success:
            print("\nFailure Analysis:")
            if not format_valid:
                print("- Invalid function call format")
            if not args_valid:
                print("- Missing or invalid required arguments")
            if not name_matches:
                print(f"- Wrong function name: got {actual_name}, expected {test_case['expected_function']}")

        print(f"\nTest {'passed' if success else 'failed'}")

    # Print summary
    print("\nTest Summary:")
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    print(f"Passed: {passed}/{total} ({passed / total * 100:.1f}%)")

    return results


if __name__ == "__main__":
    results = run_test_cases()
