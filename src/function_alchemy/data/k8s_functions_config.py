"""
Configuration file containing Kubernetes function definitions using OpenAI's function specification.
"""

K8S_FUNCTIONS = [
    # Existing functions with more variations
    {
        "name": "get_number_of_nodes",
        "description": "Get the number of nodes in the Kubernetes cluster",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "variations": [
            {
                "instruction": "How many nodes are in my cluster?",
                "thought": "I need to check the total number of nodes in the cluster. The get_number_of_nodes function will provide this information.",
            },
            {
                "instruction": "Tell me the node count in my Kubernetes environment",
                "thought": "To find the total node count in the Kubernetes environment, I should use the get_number_of_nodes function.",
            },
            {
                "instruction": "What's my cluster size in terms of nodes?",
                "thought": "To determine the cluster size in terms of nodes, I'll use the get_number_of_nodes function.",
            },
        ],
    },
    {
        "name": "get_number_of_nodes",
        "description": "Get the number of nodes in the Kubernetes cluster",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_number_of_pods",
        "description": "Get the number of pods in the Kubernetes cluster",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "variations": [
            {
                "instruction": "How many pods are currently running in my Kubernetes cluster?",
                "thought": "To find the number of running pods, I should use the get_number_of_pods function which will return the total count across all namespaces.",
            },
            {
                "instruction": "Count all pods in the cluster",
                "thought": "To count all pods in the cluster, I'll use the get_number_of_pods function to get the total across all namespaces.",
            },
            {
                "instruction": "Show me the total pod count",
                "thought": "To show the total pod count, I need to use the get_number_of_pods function.",
            },
            {
                "instruction": "What's my total pod count across all namespaces?",
                "thought": "To get the total pod count across all namespaces, I'll use the get_number_of_pods function.",
            },
        ],
    },
    {
        "name": "analyze_deployment_logs",
        "description": "Analyze logs from all pods in a deployment",
        "parameters": {
            "type": "object",
            "properties": {
                "deployment_name": {"type": "string", "description": "Name of the deployment to analyze"},
                "namespace": {"type": "string", "description": "Kubernetes namespace where the deployment is located"},
            },
            "required": ["deployment_name"],
        },
        "variations": [
            {
                "instruction": "Check the logs of my 'frontend' deployment for any errors",
                "thought": "I need to analyze the logs from the frontend deployment to look for errors. I'll use analyze_deployment_logs function with the deployment name.",
                "arguments": {"deployment_name": "frontend", "namespace": "default"},
            },
            {
                "instruction": "Show me the logs from the backend deployment in the production namespace",
                "thought": "To view the logs from the backend deployment in production, I'll use the analyze_deployment_logs function with the specific deployment and namespace.",
                "arguments": {"deployment_name": "backend", "namespace": "production"},
            },
            {
                "instruction": "Get all logs from the auth-service deployment",
                "thought": "To retrieve all logs from the auth-service deployment, I'll use the analyze_deployment_logs function. Since no namespace is specified, I'll use the default namespace.",
                "arguments": {"deployment_name": "auth-service", "namespace": "default"},
            },
            {
                "instruction": "I need to see the logs for my database deployment in the db namespace",
                "thought": "To view the logs for the database deployment in the db namespace, I'll use the analyze_deployment_logs function with the appropriate parameters.",
                "arguments": {"deployment_name": "database", "namespace": "db"},
            },
        ],
    },
    {
        "name": "get_version_info",
        "description": "Get version information for both Kubernetes API server and nodes",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_cluster_name",
        "description": "Get the name of the current Kubernetes cluster",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_cluster_status",
        "description": "Get detailed status information about the Kubernetes cluster",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "variations": [
            {
                "instruction": "Give me an overview of my cluster's health",
                "thought": (
                    "To check the cluster's health, I should use get_cluster_status which "
                    "will provide information about nodes and pods."
                ),
            }
        ],
    },
]

# Additional example data for general function calls using OpenAI spec
GENERAL_EXAMPLES = [
    {
        "instruction": "What's the weather in London?",
        "thought": "I need to use the weather tool to get the forecast.",
        "functions": [
            {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string", "description": "The city to get weather for"}},
                    "required": ["location"],
                },
            }
        ],
        "output": {"function_call": {"name": "get_weather", "arguments": {"location": "London"}}},
    },
    {
        "instruction": "What's the current time in Tokyo",
        "thought": "I need to use the time tool to get the current time.",
        "functions": [
            {
                "name": "get_time",
                "description": "Get the current time in a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string", "description": "The city to get time for"}},
                    "required": ["location"],
                },
            }
        ],
        "output": {"function_call": {"name": "get_time", "arguments": {"location": "Tokyo"}}},
    },
]

# Updated training prompt template for OpenAI function calling format
PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with available functions and additional context. Write a response that appropriately completes the request.

### Instruction:
You are a helpful assistant capable of completing tasks including function calls.

### Available Functions:
{functions}

### Question:
{instruction}

### Response:
<think>
{thought}
</think>
{output}"""
