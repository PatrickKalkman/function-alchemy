import anthropic
import json
from typing import List, Dict
import time
from pathlib import Path
from dotenv import load_dotenv
import os
from datetime import datetime
import sys

load_dotenv()


class TrainingDataGenerator:
    def __init__(self, anthropic_api_key: str):
        self.client = anthropic.Client(api_key=anthropic_api_key)
        self.start_time = time.time()

    def generate_variations(
        self,
        function_def: Dict,
        base_examples: List[Dict],
        num_variations: int = 10,
        batch_number: int = 1,
        total_batches: int = 1,
    ) -> List[Dict]:
        batch_start = time.time()
        print(f"\n⏳ Starting batch {batch_number}/{total_batches}")

        prompt = f"""You must respond with ONLY a raw JSON array. No other text.
The array should contain {num_variations} different ways to request this K8s operation:

Function: {json.dumps(function_def, indent=2)}
Example: {json.dumps(base_examples[0], indent=2)}

Format each variation exactly like this:
{{"messages": [
    {{"role": "user", "content": "request"}},
    {{"role": "assistant", "function_call": {{"name": "{function_def["name"]}", "arguments": {{...}}}}}}
]}}

Include a mix of:
- Different expertise levels
- Various contexts (urgent vs routine)
- Different phrasings
Remember: Respond with ONLY the JSON array, nothing additional characters nothing else."""

        try:
            print("🔄 Generating... (each . is progress)")

            # Here's the key fix - we handle the streaming properly
            full_response = ""
            chunks_received = 0
            with self.client.messages.stream(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0.7,
                system="You are helping create training data for a Kubernetes operations tool.",
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for chunk in stream:
                    if chunk.type == "content_block_delta":
                        chunks_received += 1
                        # Get the text content from the delta
                        full_response += chunk.delta.text
                        if chunks_received % 10 == 0:
                            elapsed = time.time() - batch_start
                            print(f"\r💭 Thinking... {chunks_received} chunks ({elapsed:.1f}s)", end="")

            print("\n✨ Processing response...")

            # Clean up the response and find the JSON array
            full_response = full_response.strip()

            # If response doesn't start with [, look for it
            if not full_response.startswith("["):
                start = full_response.find("[")
                if start == -1:
                    print("❌ No JSON array found in response")
                    return []
                full_response = full_response[start:]

            # If response doesn't end with ], find the last one
            if not full_response.endswith("]"):
                end = full_response.rfind("]")
                if end == -1:
                    print("❌ No closing bracket found")
                    return []
                full_response = full_response[: end + 1]

            json_str = full_response

            try:
                variations = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing error: {str(e)}")
                print(json_str)
                return []

            # Validate the variations
            print("🔍 Validating...", end="", flush=True)
            validated = []
            for var in variations:
                if self._validate_variation(var, function_def):
                    validated.append(var)
                print(".", end="", flush=True)

            batch_time = time.time() - batch_start
            print(f"\n✅ Batch complete in {batch_time:.1f}s")

            if validated:
                print(f"📊 Success rate: {len(validated)}/{len(variations)} variations")
                print(f"💡 Example: '{validated[0]['messages'][0]['content']}'")

            return validated

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print(f"🔍 Error details: {type(e).__name__}")
            return []

    def _validate_variation(self, variation: Dict, function_def: Dict) -> bool:
        try:
            # Basic structure checks
            if "messages" not in variation:
                return False

            messages = variation["messages"]
            if len(messages) != 2:
                return False

            if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
                return False

            # Function call validation
            func_call = messages[1].get("function_call")
            if not func_call:
                return False

            if func_call["name"] != function_def["name"]:
                return False

            # Arguments validation
            required_params = function_def["parameters"].get("required", [])
            args = (
                json.loads(func_call["arguments"])
                if isinstance(func_call["arguments"], str)
                else func_call["arguments"]
            )

            for param in required_params:
                if param not in args:
                    return False

            return True

        except Exception:
            return False

    def generate_all_training_data(
        self, functions_file: str, examples_file: str, output_file: str, variations_per_function: int = 50
    ):
        print("\n🎬 Starting generation process...")
        self.start_time = time.time()

        # Load configuration
        with open(functions_file, "r") as f:
            functions = json.load(f)["functions"]
        with open(examples_file, "r") as f:
            training_examples = json.load(f)["training_examples"]

        print(f"📋 Loaded {len(functions)} functions and {len(training_examples)} examples")

        all_variations = []
        batch_size = 10

        for idx, function in enumerate(functions, 1):
            function_start = time.time()
            print(f"\n🔧 Function {idx}/{len(functions)}: {function['name']}")

            base_examples = [
                ex for ex in training_examples if ex["messages"][1]["function_call"]["name"] == function["name"]
            ]

            if not base_examples:
                print(f"⚠️ No examples found for {function['name']}, skipping...")
                continue

            remaining = variations_per_function
            total_batches = (remaining + batch_size - 1) // batch_size
            current_batch = 1
            function_variations = []

            while remaining > 0:
                current_size = min(batch_size, remaining)

                variations = self.generate_variations(
                    function,
                    base_examples,
                    num_variations=current_size,
                    batch_number=current_batch,
                    total_batches=total_batches,
                )

                function_variations.extend(variations)
                remaining -= current_size
                current_batch += 1

                if remaining > 0:
                    print("\n😴 Brief pause...")
                    time.sleep(1)

            all_variations.extend(function_variations)
            function_time = time.time() - function_start
            print(f"\n✨ Generated {len(function_variations)} variations in {function_time:.1f}s")

        # Save all variations
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump({"training_examples": all_variations}, f, indent=2)

        total_time = time.time() - self.start_time
        print(f"\n🎉 All done! Generated {len(all_variations)} variations")
        print(f"⏱️ Total time: {total_time:.1f}s")
        print(f"💾 Saved to: {output_file}")


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    FUNCTIONS_PATH = os.path.join(SCRIPT_DIR, "functions", "cluster_operations.json")
    EXAMPLES_PATH = os.path.join(SCRIPT_DIR, "training", "training_examples.json")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_PATH = os.path.join(SCRIPT_DIR, "generated", f"training_data_{timestamp}.json")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    if not ANTHROPIC_API_KEY:
        print("❌ No ANTHROPIC_API_KEY found in environment!")
        sys.exit(1)

    generator = TrainingDataGenerator(ANTHROPIC_API_KEY)
    generator.generate_all_training_data(
        functions_file=FUNCTIONS_PATH, examples_file=EXAMPLES_PATH, output_file=OUTPUT_PATH, variations_per_function=50
    )
