#!/usr/bin/env python3
"""Strip all docstrings from Python files while preserving code"""

import ast
import sys
from pathlib import Path


class DocstringRemover(ast.NodeTransformer):
    """Remove docstrings from AST"""

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if (node.body and
            isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            node.body.pop(0)
        return node

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        if (node.body and
            isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            node.body.pop(0)
        return node

    def visit_Module(self, node):
        self.generic_visit(node)
        if (node.body and
            isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            node.body.pop(0)
        return node


def strip_docstrings_from_file(filepath: Path) -> bool:
    """Remove docstrings from a Python file, return True if modified"""

    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        print(f"Skipping {filepath} - syntax error")
        return False

    remover = DocstringRemover()
    new_tree = remover.visit(tree)

    new_source = ast.unparse(new_tree)

    if new_source != source:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_source)
        print(f"Stripped {filepath}")
        return True

    return False


def main():
    slime_dir = Path(__file__).parent / 'slime'

    if not slime_dir.exists():
        print(f"Error: {slime_dir} not found")
        sys.exit(1)

    py_files = list(slime_dir.rglob('*.py'))
    print(f"Found {len(py_files)} Python files in slime/")
    print()

    modified = 0
    for filepath in py_files:
        if strip_docstrings_from_file(filepath):
            modified += 1

    print()
    print(f"Modified {modified} files")
    print(f"Unchanged {len(py_files) - modified} files")


if __name__ == '__main__':
    main()
