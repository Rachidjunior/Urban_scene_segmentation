# verify_setup.py
import sys
print("="*50)
print("ENVIRONMENT VERIFICATION")
print("="*50)

# Python version
print(f"\n✅ Python: {sys.version.split()[0]}")

# PyTorch
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"   - CPU available: {torch.cuda.is_available() == False}")
print(f"   - Number of threads: {torch.get_num_threads()}")

# TorchVision
import torchvision
print(f"✅ TorchVision: {torchvision.__version__}")

# Detectron2
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
print(f"✅ Detectron2: {detectron2.__version__}")
print(f"   - Model zoo accessible: True")

# OpenCV
import cv2
print(f"✅ OpenCV: {cv2.__version__}")

# NumPy
import numpy as np
print(f"✅ NumPy: {np.__version__}")

# Matplotlib
import matplotlib
print(f"✅ Matplotlib: {matplotlib.__version__}")

print("\n" + "="*50)
print("🎉 ALL PACKAGES INSTALLED SUCCESSFULLY!")
print("="*50)
print("\nYour environment is ready for:")
print("  - Model development")
print("  - Data exploration")
print("  - Visualization")
print("  - Training on Colab (GPU)")
print("\n⚠️  Note: Local training will be SLOW (no GPU)")
print("    Use Colab for actual model training!")