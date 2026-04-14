"""Remove broken torch installation files from site-packages."""
import shutil
import sys
from pathlib import Path

site_packages = Path(sys.executable).parent / "Lib" / "site-packages"
print(f"Site-packages: {site_packages}")

dirs_to_remove = ["torch", "torchgen", "functorch"]
for d in dirs_to_remove:
    target = site_packages / d
    if target.exists():
        print(f"Removing {target}...")
        shutil.rmtree(target, ignore_errors=True)
        print(f"  Removed {d}")
    else:
        print(f"  {d} not found (OK)")

# Also clean up dist-info
for p in site_packages.glob("torch-*.dist-info"):
    print(f"Removing {p}...")
    shutil.rmtree(p, ignore_errors=True)

print("Done!")
