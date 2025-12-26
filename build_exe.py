#!/usr/bin/env python3
"""
TruthScript Build Script - Windows Only
Creates standalone .exe for Windows distribution

Usage:
    python build_exe.py
    
Note: Run from GitHub Actions for proper ffmpeg bundling
"""

import subprocess
import sys
import os
import shutil

def check_dependencies():
    """Check if required packages are installed."""
    required = ['pyinstaller', 'openai-whisper', 'torch', 'torchaudio', 'ffmpeg-python']
    missing = []
    
    for pkg in required:
        try:
            if pkg == 'openai-whisper':
                __import__('whisper')
            elif pkg == 'ffmpeg-python':
                __import__('ffmpeg')
            else:
                __import__(pkg.replace('-', '_'))
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)

def check_ffmpeg():
    """Check if ffmpeg binaries are available for bundling."""
    ffmpeg_dir = 'ffmpeg_bin'
    if os.path.isdir(ffmpeg_dir):
        files = os.listdir(ffmpeg_dir)
        print(f"Found ffmpeg_bin directory with: {files}")
        return True
    else:
        print("WARNING: ffmpeg_bin directory not found!")
        print("The .exe will require users to install ffmpeg separately.")
        print("For standalone builds, run via GitHub Actions workflow.")
        return False

def clean_build():
    """Clean previous build artifacts."""
    dirs_to_clean = ['build', 'dist', '__pycache__']
    for d in dirs_to_clean:
        if os.path.exists(d):
            print(f"Cleaning {d}...")
            shutil.rmtree(d)

def build():
    """Build the standalone Windows executable using PyInstaller."""
    print("=" * 60)
    print("TruthScript Windows Build")
    print("=" * 60)
    
    check_dependencies()
    has_ffmpeg = check_ffmpeg()
    
    import PyInstaller
    print(f"PyInstaller version: {PyInstaller.__version__}")
    
    # Clean previous builds
    clean_build()
    
    # Build using the spec file
    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        "whisper_app.spec",
        "--clean",
        "--noconfirm"
    ]
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    try:
        subprocess.check_call(cmd)
        print("\n" + "=" * 60)
        print("BUILD SUCCESSFUL!")
        print("=" * 60)
        
        print("\nThe executable is located at:")
        print("  dist\\TruthScript\\TruthScript.exe")
        print("\nTo distribute:")
        print("  1. Copy the entire 'dist\\TruthScript' folder")
        print("  2. Users double-click TruthScript.exe to run")
        
        if has_ffmpeg:
            print("\nFFmpeg: BUNDLED (standalone operation)")
        else:
            print("\nFFmpeg: NOT BUNDLED (users must install separately)")
        
        print("\nSpeed optimizations enabled:")
        print("  - CUDA GPU acceleration (if NVIDIA GPU available)")
        print("  - FP16 half-precision on GPU")
        print("  - Multi-threaded CPU fallback")
        print("\nNOTE: First run will download the Whisper model (~140MB for base)")
        
    except subprocess.CalledProcessError as e:
        print(f"\nBuild failed with error: {e}")
        print("\nTroubleshooting tips:")
        print("  1. Make sure all dependencies are installed")
        print("  2. Try: pip install --upgrade pyinstaller")
        print("  3. Check for antivirus blocking PyInstaller")
        sys.exit(1)

if __name__ == "__main__":
    build()
