# ========================================
# BUILD DETECTRON2 WHEEL FOR KAGGLE
# ========================================
# Run this Python script locally to build a Detectron2 wheel
# Then upload the wheel to Kaggle as a dataset

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run shell command with output"""
    print(f"\n{'='*70}")
    print(f"🔄 {description}")
    print(f"{'='*70}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        sys.exit(1)
    
    print(f"✅ Success: {description}")
    return result

def main():
    print("="*70)
    print("BUILDING DETECTRON2 WHEEL FOR KAGGLE")
    print("="*70)
    
    # Create build directory
    build_dir = Path("detectron2_wheel_build")
    build_dir.mkdir(exist_ok=True)
    os.chdir(build_dir)
    
    print(f"\n📁 Working in: {build_dir.absolute()}")
    
    # Install dependencies
    run_command(
        f"{sys.executable} -m pip install --upgrade pip wheel setuptools",
        "Installing build dependencies"
    )
    
    # Clone Detectron2
    if not Path("detectron2").exists():
        run_command(
            "git clone https://github.com/facebookresearch/detectron2.git",
            "Cloning Detectron2 repository"
        )
    else:
        print("\n✅ Detectron2 already cloned")
    
    # Build wheel
    os.chdir("detectron2")
    
    run_command(
        f"{sys.executable} setup.py bdist_wheel",
        "Building wheel (takes 5-10 minutes)"
    )
    
    # Find the wheel
    dist_dir = Path("dist")
    wheels = list(dist_dir.glob("*.whl"))
    
    print("\n" + "="*70)
    print("✅ BUILD COMPLETE!")
    print("="*70)
    
    if wheels:
        wheel_path = wheels[0]
        print(f"\n📦 Wheel created: {wheel_path.name}")
        print(f"📁 Location: {wheel_path.absolute()}")
        print(f"📊 Size: {wheel_path.stat().st_size / (1024*1024):.1f} MB")
        
        print("\n" + "="*70)
        print("📤 NEXT STEPS:")
        print("="*70)
        print("1. Go to https://www.kaggle.com/datasets")
        print("2. Click 'New Dataset'")
        print(f"3. Upload: {wheel_path.absolute()}")
        print("4. Name it 'detectron2-wheel'")
        print("5. In your Kaggle notebook:")
        print("   - Add this dataset")
        print("   - Run: !pip install /kaggle/input/detectron2-wheel/*.whl")
        print("="*70)
    else:
        print("\n❌ No wheel found in dist/")

if __name__ == "__main__":
    main()
