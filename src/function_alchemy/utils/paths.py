import os


def get_project_root():
    """Get absolute path to project root directory."""
    current_file = os.path.abspath(__file__)  # Get this file's path
    # Go up 4 levels: utils -> function_alchemy -> src -> project_root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
    return project_root


def get_data_path():
    """Get absolute path to data directory."""
    return os.path.join(get_project_root(), "src", "function_alchemy", "data", "training")
