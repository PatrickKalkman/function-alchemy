"""
Generator for creating training data from Kubernetes function definitions using OpenAI function spec.
"""

import json
from typing import List, Dict, Any
from function_alchemy.data.k8s_functions_config import K8S_FUNCTIONS, GENERAL_EXAMPLES


def create_training_example(
    instruction: str, thought: str, output: Dict[str, Any], available_functions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Create a training example in the required format."""
    return {"instruction": instruction, "thought": thought, "output": output, "functions": available_functions}


def generate_function_call(name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generate a formatted function call in OpenAI format."""
    return {"function_call": {"name": name, "arguments": arguments or {}}}


def generate_k8s_training_data() -> List[Dict[str, Any]]:
    """Generate training data from Kubernetes functions."""
    training_data = []

    # Generate examples for each Kubernetes function
    for func in K8S_FUNCTIONS:
        # Basic query example
        instruction = f"Can you {func['description'].lower()}?"
        thought = f"I need to use the Kubernetes tools to {func['description'].lower()}. I should use the {func['name']} function."

        # Generate default example
        if not func["parameters"]["properties"]:
            output = generate_function_call(func["name"])
        else:
            # Example with default parameters
            args = {param: f"example-{param}" for param in func["parameters"]["properties"].keys()}
            output = generate_function_call(func["name"], args)

        # Create list of available functions for this example
        available_functions = [func]

        training_data.append(create_training_example(instruction, thought, output, available_functions))

        # Add variations if they exist
        if "variations" in func:
            for variation in func["variations"]:
                output = generate_function_call(
                    func["name"], variation.get("arguments", {}) if func["parameters"]["properties"] else {}
                )
                training_data.append(
                    create_training_example(variation["instruction"], variation["thought"], output, available_functions)
                )

    return training_data


def get_combined_training_data() -> List[Dict[str, Any]]:
    """Get combined training data including both K8s and general examples."""
    k8s_data = generate_k8s_training_data()
    return GENERAL_EXAMPLES + k8s_data


if __name__ == "__main__":
    # Generate the training data
    training_data = get_combined_training_data()

    # Print statistics and sample
    print(f"Generated {len(training_data)} training examples")
    print(f"- Kubernetes examples: {len(generate_k8s_training_data())}")
    print(f"- General examples: {len(GENERAL_EXAMPLES)}")

    print("\nSample training example:")
    print(json.dumps(training_data[0], indent=2))

    # Save to file
    with open("training_data.json", "w") as f:
        json.dump(training_data, f, indent=2)
