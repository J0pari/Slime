import ast
import sys
from pathlib import Path

class DocstringRemover(ast.NodeTransformer):

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant)):
            node.body.pop(0)
        if not node.body:
            node.body = [ast.Pass()]
        return node

    def visit_AsyncFunctionDef(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant)):
            node.body.pop(0)
        if not node.body:
            node.body = [ast.Pass()]
        return node

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant)):
            node.body.pop(0)
        if not node.body:
            node.body = [ast.Pass()]
        return node

    def visit_Module(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant)):
            node.body.pop(0)
        return node

def remove_docstrings_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    try:
        tree = ast.parse(source)
        remover = DocstringRemover()
        new_tree = remover.visit(tree)
        ast.fix_missing_locations(new_tree)
        new_source = ast.unparse(new_tree)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_source)
        print(f'Processed: {filepath}')
    except SyntaxError as e:
        print(f'Syntax error in {filepath}: {e}')
    except Exception as e:
        print(f'Error processing {filepath}: {e}')
if __name__ == '__main__':
    repo_root = Path('c:/Slime')
    py_files = list(repo_root.rglob('*.py'))
    for py_file in py_files:
        if '.venv' not in str(py_file) and '__pycache__' not in str(py_file):
            remove_docstrings_from_file(py_file)
    print(f'\nProcessed {len(py_files)} Python files')