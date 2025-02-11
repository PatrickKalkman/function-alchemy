import os
import json
from ..utils.paths import get_data_path

# In loader.py
PROMPT_TEMPLATE = """<|im_start|>system<|im_sep|>You are an AI assistant that helps users interact with Kubernetes clusters through function calls. When given an instruction, you should respond with an appropriate function call from the available functions.

Available Functions:
{functions}<|im_end|>
<|im_start|>user<|im_sep|>{instruction}<|im_end|>
<|im_start|>assistant<|im_sep|>{output}<|im_end|>"""


def load_training_data():
    """Load and combine training data from JSON files."""
    data_dir = get_data_path()

    # Load function definitions
    functions_path = os.path.join(data_dir, "cluster_operations.json")
    with open(functions_path, "r") as f:
        functions_data = json.load(f)

    # Load training examples
    examples_path = os.path.join(data_dir, "training_examples.json")
    with open(examples_path, "r") as f:
        examples_data = json.load(f)

    # Process training examples into required format
    processed_data = []
    for example in examples_data["training_examples"]:
        processed_data.append(
            {
                "instruction": example["messages"][0]["content"],
                "output": example["messages"][1]["function_call"],
                "functions": functions_data["functions"],
                "thought": "",  # We don't have thoughts in current data
            }
        )

    return processed_data
