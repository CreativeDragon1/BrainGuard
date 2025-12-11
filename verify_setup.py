#!/usr/bin/env python3
"""
Quick verification script to check if everything is set up correctly
"""
import os
import sys
from pathlib import Path

def check(condition, message):
    if condition:
        print(f"✓ {message}")
        return True
    else:
        print(f"✗ {message}")
        return False

print("=" * 60)
print("BrainGuard Setup Verification")
print("=" * 60)

project_root = Path(__file__).parent
all_good = True

# Check 1: Model file exists
model_path = project_root / "models" / "best_resnet.pth"
all_good &= check(model_path.exists(), f"Model file exists: {model_path.name}")
if model_path.exists():
    size_mb = model_path.stat().st_size / 1024 / 1024
    print(f"  → Size: {size_mb:.1f} MB")

# Check 2: Required folders
required_dirs = [
    "models",
    "app",
    "templates",
    "static",
    "static/uploads",
]
for dir_name in required_dirs:
    dir_path = project_root / dir_name
    all_good &= check(dir_path.exists(), f"Directory exists: {dir_name}/")

# Check 3: Required files
required_files = [
    "app/app.py",
    "models/cnn_model.py",
    "models/grad_cam.py",
    "models/preprocessing.py",
    "templates/index.html",
    "train_colab.ipynb",
    "SETUP_AND_TEST.md",
]
for file_name in required_files:
    file_path = project_root / file_name
    all_good &= check(file_path.exists(), f"File exists: {file_name}")

# Check 4: Test imports
print("\nTesting Python imports...")
try:
    import torch
    check(True, f"PyTorch installed (v{torch.__version__})")
except ImportError:
    check(False, "PyTorch installed")
    all_good = False

try:
    import flask
    check(True, f"Flask installed (v{flask.__version__})")
except ImportError:
    check(False, "Flask installed")
    all_good = False

try:
    sys.path.insert(0, str(project_root))
    from models.cnn_model import ResNetModel
    check(True, "Model classes can be imported")
except Exception as e:
    check(False, f"Model classes can be imported: {e}")
    all_good = False

# Check 5: App.py can be imported
print("\nTesting app configuration...")
try:
    os.chdir(str(project_root))
    sys.path.insert(0, str(project_root))
    # Just check if it exists and has key content
    with open(project_root / "app" / "app.py") as f:
        content = f.read()
        has_flask = "Flask" in content
        has_model_load = "load_model" in content
        has_predict = "@app.route('/predict'" in content
        all_good &= check(has_flask and has_model_load and has_predict, 
                         "App has required endpoints and functions")
except Exception as e:
    check(False, f"App validation: {e}")
    all_good = False

# Summary
print("\n" + "=" * 60)
if all_good:
    print("✓ All checks passed! Ready to run the app")
    print("\nNext steps:")
    print("1. Make sure best_resnet.pth is in models/ folder")
    print("2. Run: python app/app.py")
    print("3. Open: http://localhost:8000")
else:
    print("✗ Some checks failed. Please fix the issues above.")
    
print("=" * 60)
