#!/usr/bin/env python3
"""
This script updates the build process to include FFmpeg with the application bundle.
"""
import os
import sys
import subprocess
import shutil
import importlib.util
import importlib.machinery
from pathlib import Path

def main():
    # First, download FFmpeg
    print("Step 1: Downloading FFmpeg...")
    if not os.path.exists("download_ffmpeg.py"):
        print("ERROR: download_ffmpeg.py script not found")
        return False
    
    result = subprocess.run([sys.executable, "download_ffmpeg.py"], check=False)
    if result.returncode != 0:
        print("ERROR: Failed to download FFmpeg")
        return False
    
    # Verify the bin directory and FFmpeg binary exist
    bin_dir = Path("bin")
    if not bin_dir.exists() or not bin_dir.is_dir():
        print(f"ERROR: bin directory not found at {bin_dir}")
        return False
    
    ffmpeg_bin = bin_dir / "ffmpeg"
    if not ffmpeg_bin.exists():
        print(f"ERROR: FFmpeg binary not found at {ffmpeg_bin}")
        return False
    
    # Copy our ffmpeg_path.py hook to ensure it's included
    if not os.path.exists("ffmpeg_path.py"):
        print("ERROR: ffmpeg_path.py hook not found")
        return False
    
    # Clean output directories manually to avoid issues
    print("Cleaning output directories...")
    for dir_name in ["build", "dist"]:
        if os.path.exists(dir_name):
            try:
                shutil.rmtree(dir_name)
            except Exception as e:
                print(f"Warning: Could not remove {dir_name}: {e}")
                # Try with force
                try:
                    subprocess.run(["rm", "-rf", dir_name], check=False)
                except:
                    pass
    
    # Remove spec files
    for spec_file in Path(".").glob("*.spec"):
        try:
            os.remove(spec_file)
        except Exception as e:
            print(f"Warning: Could not remove {spec_file}: {e}")
    
    # Find required data files
    data_files = []
    
    # Add bin directory
    data_files.append((f"bin{os.pathsep}bin"))
    
    # Find lightning_fabric version.info file
    try:
        # Check if lightning_fabric is installed
        if importlib.util.find_spec("lightning_fabric") is not None:
            # Get the package location
            pkg_spec = importlib.util.find_spec("lightning_fabric")
            package_dir = os.path.dirname(pkg_spec.origin)
            
            # Look for version.info file
            version_info_path = os.path.join(package_dir, "version.info")
            if os.path.exists(version_info_path):
                # Create target directory structure
                target_dir = "lightning_fabric"
                data_spec = f"{version_info_path}{os.pathsep}{target_dir}"
                data_files.append(data_spec)
                print(f"Found lightning_fabric/version.info at: {version_info_path}")
            else:
                # Create an empty version.info file if it doesn't exist
                print("lightning_fabric/version.info not found, creating a placeholder")
                placeholder_path = os.path.join(os.getcwd(), "lightning_fabric_version.info")
                os.makedirs(os.path.dirname(placeholder_path), exist_ok=True)
                with open(placeholder_path, 'w') as f:
                    f.write("0.0.0\n")
                data_files.append(f"{placeholder_path}{os.pathsep}lightning_fabric")
    except Exception as e:
        print(f"Warning: Could not handle lightning_fabric version.info: {e}")
    
    # Now run pyinstaller directly with our parameters
    print("\nStep 2: Building the application with bundled FFmpeg...")
    
    # Prepare data files arguments
    data_args = []
    for data_spec in data_files:
        data_args.extend(["--add-data", data_spec])
    
    build_command = [
        "pyinstaller",
        "--name", "Audio Processing App",
        "--onedir",
        "--windowed",
        "--clean",
        # Add our runtime hooks
        "--runtime-hook", "ffmpeg_path.py",
        "--runtime-hook", "wx_path.py",
    ] + data_args + [
        # Main script
        "main.py"
    ]
    
    print(f"Running command: {' '.join(build_command)}")
    result = subprocess.run(build_command, check=False)
    if result.returncode != 0:
        print("ERROR: Build failed")
        return False
    
    print("\nBuild completed successfully with bundled FFmpeg!")
    print("Your app should now be able to run without requiring FFmpeg to be installed separately.")
    return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1) 