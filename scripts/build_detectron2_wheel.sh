#!/bin/bash
# ========================================
# BUILD DETECTRON2 WHEEL FOR KAGGLE
# ========================================
# Run this script locally, then upload the wheel to Kaggle as a dataset

echo "=========================================="
echo "Building Detectron2 Wheel for Kaggle"
echo "=========================================="

# Create build directory
BUILD_DIR="detectron2_build"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Clone Detectron2
echo "📦 Cloning Detectron2..."
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2

# Build wheel
echo "🔨 Building wheel (this takes 5-10 minutes)..."
python -m pip install wheel
python setup.py bdist_wheel

# The wheel will be in dist/
echo ""
echo "=========================================="
echo "✅ BUILD COMPLETE!"
echo "=========================================="
echo "Wheel location: $(pwd)/dist/"
ls -lh dist/

echo ""
echo "📤 NEXT STEPS:"
echo "1. Upload the .whl file from dist/ to Kaggle as a dataset"
echo "2. Name it 'detectron2-wheel'"
echo "3. In your notebook, add this dataset"
echo "4. Install with: !pip install /kaggle/input/detectron2-wheel/detectron2-*.whl"
echo "=========================================="
