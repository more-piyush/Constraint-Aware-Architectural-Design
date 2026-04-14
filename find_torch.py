"""Find torch installation location."""
import sys
print(f"Python: {sys.executable}")
print(f"Path:")
for p in sys.path:
    print(f"  {p}")

try:
    import torch
    print(f"\ntorch found at: {torch.__file__}")
except ImportError as e:
    print(f"\ntorch import failed: {e}")

# Check if torch directory exists anywhere on sys.path
from pathlib import Path
for p in sys.path:
    pp = Path(p)
    if pp.exists():
        torch_dir = pp / "torch"
        if torch_dir.exists():
            print(f"\ntorch DIRECTORY found at: {torch_dir}")
            # List a few files
            files = list(torch_dir.iterdir())[:10]
            for f in files:
                print(f"  {f.name}")
