"""
ast_validator.py — Python syntax validation using the ast module
================================================================
Every generated code snippet is checked for valid Python syntax.
Invalid code is rejected and regenerated with different parameters.

Python's built-in ast module parses code into an Abstract Syntax Tree.
If parsing fails, the code has a syntax error.
"""

import ast
from typing import Optional


def is_valid_python(code: str) -> bool:
    """Return True if code is syntactically valid Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def get_syntax_error(code: str) -> Optional[str]:
    """Return the syntax error message, or None if code is valid."""
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"Line {e.lineno}: {e.msg}"


def extract_functions(code: str) -> list[str]:
    """Extract all function names defined in the code."""
    try:
        tree = ast.parse(code)
        return [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        ]
    except SyntaxError:
        return []


if __name__ == "__main__":
    valid_code   = "for i in range(10):\n    print(i)"
    invalid_code = "for i in range(10)\n    print(i)"   # missing colon

    print(f"Valid   : {is_valid_python(valid_code)}")
    print(f"Invalid : {is_valid_python(invalid_code)}")
    print(f"Error   : {get_syntax_error(invalid_code)}")
