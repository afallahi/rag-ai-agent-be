import os
import sys

# Force ASCII-friendly output
sys.stdout.reconfigure(encoding='utf-8')

# Folders to ignore
EXCLUDE_DIRS = {".git", ".vscode", "__pycache__", ".pytest_cache", ".idea", "debug_chunks"}

def print_tree(root_dir, prefix=""):
    try:
        entries = sorted(os.listdir(root_dir))
    except PermissionError:
        return
    
    # Filter out unwanted directories/files
    entries = [e for e in entries if e not in EXCLUDE_DIRS]

    for i, entry in enumerate(entries):
        path = os.path.join(root_dir, entry)
        connector = "`-- " if i == len(entries) - 1 else "|-- "
        print(prefix + connector + entry)
        if os.path.isdir(path):
            extension = "    " if i == len(entries) - 1 else "|   "
            print_tree(path, prefix + extension)

if __name__ == "__main__":
    print(".")
    print_tree(".")
