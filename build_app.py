#!/usr/bin/env python3
import os
import sys
import shutil
import platform
import subprocess
import importlib.util
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

def find_package_data_files():
    """Find package data files that need to be included with the application."""
    additional_data = []
    
    # Find lightning_fabric version.info
    try:
        # Check if lightning_fabric is installed
        if importlib.util.find_spec("lightning_fabric") is not None:
            # Get the package location
            package_path = importlib.util.find_spec("lightning_fabric").origin
            package_dir = os.path.dirname(package_path)
            
            # Look for version.info file
            version_info_path = os.path.join(package_dir, "version.info")
            if os.path.exists(version_info_path):
                # Format properly for cross-platform compatibility
                sep = ";" if platform.system() == "Windows" else ":"
                target_dir = os.path.join("lightning_fabric")
                data_spec = f"{version_info_path}{sep}{target_dir}"
                additional_data.append(data_spec)
                print(f"Adding lightning_fabric/version.info: {data_spec}")
            else:
                print("WARNING: lightning_fabric/version.info not found")
    except Exception as e:
        print(f"WARNING: Could not locate lightning_fabric/version.info: {e}")
    
    # Find speechbrain directories and files
    try:
        # Check if speechbrain is installed
        if importlib.util.find_spec("speechbrain") is not None:
            # Get the package location
            package_path = importlib.util.find_spec("speechbrain").origin
            package_dir = os.path.dirname(package_path)
            
            # List of subdirectories to include
            speechbrain_dirs = ["utils", "dataio"]
            
            # Add version.txt file
            version_txt_path = os.path.join(package_dir, "version.txt")
            if os.path.exists(version_txt_path):
                sep = ";" if platform.system() == "Windows" else ":"
                target_dir = "speechbrain"
                data_spec = f"{version_txt_path}{sep}{target_dir}"
                additional_data.append(data_spec)
                print(f"Adding speechbrain/version.txt: {data_spec}")
            else:
                print("WARNING: speechbrain/version.txt not found")
            
            # Add directories
            for subdir in speechbrain_dirs:
                dir_path = os.path.join(package_dir, subdir)
                if os.path.exists(dir_path) and os.path.isdir(dir_path):
                    # Format properly for cross-platform compatibility
                    sep = ";" if platform.system() == "Windows" else ":"
                    target_dir = f"speechbrain/{subdir}"
                    data_spec = f"{dir_path}{sep}{target_dir}"
                    additional_data.append(data_spec)
                    print(f"Adding speechbrain/{subdir} directory: {data_spec}")
                else:
                    print(f"WARNING: speechbrain/{subdir} directory not found")
    except Exception as e:
        print(f"WARNING: Could not locate speechbrain directories or files: {e}")
    
    return additional_data

def build_macos():
    """Build macOS application bundle."""
    print("Building macOS application...")
    
    # Find required data files
    additional_data = find_package_data_files()
    
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
    
    # Add data files
    for data_spec in additional_data:
        cmd.extend(["--add-data", data_spec])
    
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
    
    # Find required data files
    additional_data = find_package_data_files()
    
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
    
    # Add data files
    for data_spec in additional_data:
        cmd.extend(["--add-data", data_spec])
    
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