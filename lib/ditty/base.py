"""
Base class for all ditty pipeline components.
"""
import re


def camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case, handling acronyms properly."""
    # Handle acronyms followed by lowercase (e.g., MSELoss -> mse_loss)
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    # Handle lowercase followed by uppercase (e.g., camelCase -> camel_case)
    name = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', name)
    return name.lower()


class DittyBase:
    """Base class providing contract and name for all pipeline components."""

    def __init__(self, name: str = "", contract: str = ""):
        self.name = name or camel_to_snake(self.__class__.__name__)
        self.contract = contract
