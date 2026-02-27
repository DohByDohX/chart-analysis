import ast
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def check_torch_load_security(filepath):
    """
    Checks if torch.load calls in the given file have weights_only=True.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filepath)
    except Exception as e:
        logging.error(f"Failed to parse {filepath}: {e}")
        return False

    vulnerable = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "load":
                # Check if it's torch.load
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "torch":
                    # Check arguments
                    has_weights_only = False
                    weights_only_value = None

                    for keyword in node.keywords:
                        if keyword.arg == "weights_only":
                            has_weights_only = True
                            if isinstance(keyword.value, ast.Constant):
                                weights_only_value = keyword.value.value
                            elif isinstance(keyword.value, ast.NameConstant): # Python < 3.8
                                weights_only_value = keyword.value.value
                            break

                    if not has_weights_only:
                        logging.error(f"Vulnerability found in {filepath} at line {node.lineno}: torch.load called without weights_only argument.")
                        vulnerable = True
                    elif weights_only_value is not True:
                         logging.error(f"Vulnerability found in {filepath} at line {node.lineno}: torch.load called with weights_only={weights_only_value}. It must be True.")
                         vulnerable = True

    if not vulnerable:
        logging.info(f"No vulnerabilities found in {filepath}.")
        return True
    else:
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_security.py <file_path>")
        sys.exit(1)

    filepath = sys.argv[1]
    if check_torch_load_security(filepath):
        sys.exit(0)
    else:
        sys.exit(1)
