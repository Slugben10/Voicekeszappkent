#!/usr/bin/env python3
import os
import sys
import shutil
import platform
import subprocess
from pathlib import Path

# Application name and output directory
APP_NAME = "Audio Processing App"
OUTPUT_DIR = "dist"

def clean_output():
    """Clean the output directories."""
    print("Cleaning output directories...")
    for dir_name in ["build", "dist"]:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
    
    # Remove spec files
    for spec_file in Path(".").glob("*.spec"):
        os.remove(spec_file)

def get_icon_path():
    """Get platform-specific icon path."""
    system = platform.system()
    if system == "Darwin":  # macOS
        return "icon.icns"
    elif system == "Windows":
        return "icon.ico"
    return None

def create_directories():
    """Create directories that should be bundled with the app."""
    print("Creating required directories...")
    for dir_name in ["Documents", "Transcripts", "Summaries"]:
        os.makedirs(os.path.join(OUTPUT_DIR, dir_name), exist_ok=True)

def build_macos():
    """Build macOS application bundle."""
    print("Building macOS application...")
    
    # Create PyInstaller command
    cmd = [
        "pyinstaller",
        "--name", APP_NAME,
        "--onedir",
        "--windowed",
        "--clean",
        "--noconfirm",
    ]
    
    # Add icon if exists
    icon_path = get_icon_path()
    if icon_path and os.path.exists(icon_path):
        cmd.extend(["--icon", icon_path])
    
    # Add hidden imports for OpenAI and wxPython
    cmd.extend([
        "--hidden-import", "openai",
        "--hidden-import", "wx",
        "--hidden-import", "wx.adv",
    ])
    
    # Add main script
    cmd.append("main.py")
    
    # Run PyInstaller
    subprocess.run(cmd)
    
    # Create directories
    create_directories()
    
    print(f"macOS build completed. App is located in {OUTPUT_DIR}/{APP_NAME}.app")

def build_windows():
    """Build Windows executable."""
    print("Building Windows application...")
    
    # Create PyInstaller command
    cmd = [
        "pyinstaller",
        "--name", APP_NAME,
        "--onedir",
        "--windowed",
        "--clean",
        "--noconfirm",
    ]
    
    # Add icon if exists
    icon_path = get_icon_path()
    if icon_path and os.path.exists(icon_path):
        cmd.extend(["--icon", icon_path])
    
    # Add hidden imports for OpenAI and wxPython
    cmd.extend([
        "--hidden-import", "openai",
        "--hidden-import", "wx",
        "--hidden-import", "wx.adv",
    ])
    
    # Add main script
    cmd.append("main.py")
    
    # Run PyInstaller
    subprocess.run(cmd)
    
    # Create directories
    create_directories()
    
    print(f"Windows build completed. Executable is located in {OUTPUT_DIR}/{APP_NAME}.exe")

def main():
    """Main function to build the application."""
    # Clean output directories
    clean_output()
    
    # Determine platform and build accordingly
    system = platform.system()
    if system == "Darwin":  # macOS
        build_macos()
    elif system == "Windows":
        build_windows()
    else:
        print(f"Unsupported platform: {system}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
