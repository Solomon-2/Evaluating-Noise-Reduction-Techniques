"""
Import utility for notebooks to easily import from the src directory.
"""
import sys
import os

def setup_src_imports():
    """Add the src directory to Python path for importing custom modules."""
    # Get the absolute path to the src directory
    notebook_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    src_path = os.path.join(os.path.dirname(notebook_dir), 'src')
    
    # Add to path if not already there
    if src_path not in sys.path:
        sys.path.append(src_path)
        print(f"Added {src_path} to Python path")
    
    return src_path

# Auto-setup when imported
if __name__ != "__main__":
    setup_src_imports()
