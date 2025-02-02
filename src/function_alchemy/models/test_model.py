"""
Test script for evaluating the fine-tuned model's function calling capabilities.
"""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from function_alchemy.data.k8s_functions_config import K8S_FUNCTIONS, PROMPT_TEMPLATE


def load_model(model_path: str):
    """Load the fine-tuned model and tokenizer."""
    print(f"Loading model from {model_path}")

    # Load base model
    base_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Load the PEFT adapter
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    return model, tokenizer


def format_prompt(instruction: str, available_functions: list) -> str:
    """Format the prompt according to our template."""
    functions_str = json.dumps(available_functions, indent=2)
    return PROMPT_TEMPLATE.format(
        functions=functions_str,
        instruction=instruction,
        thought="",  # Leave empty for inference
        output="",  # Leave empty for inference
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


def parse_function_call(response: str) -> dict:
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

        # Try to find the first { and last } to extract just the JSON
        start = function_json.find("{")
        end = function_json.rfind("}") + 1
        if start == -1 or end == 0:
            return None

        function_json = function_json[start:end]

        # Parse the JSON
        parsed = json.loads(function_json)

        # Verify the required structure
        if "function_call" not in parsed:
            return None
        if "name" not in parsed["function_call"]:
            return None

        return parsed
    except (json.JSONDecodeError, KeyError):
        print(f"Failed to parse JSON: {function_json}")
        return None


def evaluate_response(response: dict, expected_name: str, required_args: list = None) -> bool:
    """Evaluate if the response matches expected format and contains required arguments."""
    if not response or "function_call" not in response:
        return False

    function_call = response["function_call"]
    if "name" not in function_call or "arguments" not in function_call:
        return False

    if function_call["name"] != expected_name:
        return False

    if required_args:
        args = function_call["arguments"]
        return all(arg in args for arg in required_args)

    return True


def run_test_cases():
    """Run a series of test cases on the fine-tuned model and provide detailed diagnostics."""
    model_path = "DeepSeek-R1-Distill-Qwen-1.5B-func-openai"
    model, tokenizer = load_model(model_path)

    test_cases = [
        {
            "instruction": "How many pods are running in the cluster?",
            "expected_function": "get_number_of_pods",
            "required_args": [],
        },
        {
            "instruction": "Check the logs of my frontend deployment in the default namespace",
            "expected_function": "analyze_deployment_logs",
            "required_args": ["deployment_name", "namespace"],
        },
        {
            "instruction": "What's the current status of my cluster?",
            "expected_function": "get_cluster_status",
            "required_args": [],
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
        print(f"Parsed function call:\n{json.dumps(function_call, indent=2)}")

        # Evaluate response
        success = evaluate_response(function_call, test_case["expected_function"], test_case["required_args"])

        results.append(
            {
                "instruction": test_case["instruction"],
                "expected_function": test_case["expected_function"],
                "actual_response": function_call,
                "success": success,
            }
        )

        if not success:
            print("\nFailure Analysis:")
            if not function_call:
                print("- Failed to parse function call JSON")
            elif "function_call" not in function_call:
                print("- Missing 'function_call' in response")
            elif "name" not in function_call["function_call"]:
                print("- Missing 'name' in function_call")
            elif function_call["function_call"]["name"] != test_case["expected_function"]:
                print(
                    f"- Wrong function name: got {function_call['function_call']['name']}, expected {test_case['expected_function']}"
                )
            elif test_case["required_args"]:
                missing_args = [
                    arg for arg in test_case["required_args"] if arg not in function_call["function_call"]["arguments"]
                ]
                if missing_args:
                    print(f"- Missing required arguments: {missing_args}")
        print(f"\nTest {'passed' if success else 'failed'}")

    # Print summary
    print("\nTest Summary:")
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    print(f"Passed: {passed}/{total} ({passed / total * 100:.1f}%)")

    return results


if __name__ == "__main__":
    results = run_test_cases()
