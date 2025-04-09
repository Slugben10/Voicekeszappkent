#!/usr/bin/env python3
import os
import sys
import platform
import subprocess
import tempfile
import shutil
import zipfile
import requests
from pathlib import Path

def main():
    """Download FFmpeg for the current platform and place it in the correct location for bundling."""
    print("Downloading FFmpeg for bundling with the application...")
    
    # Create a directory for binaries if it doesn't exist
    bin_dir = Path("bin")
    bin_dir.mkdir(exist_ok=True)
    
    # Download the appropriate FFmpeg binary based on platform
    system = platform.system()
    machine = platform.machine()
    
    if system == "Darwin":  # macOS
        if machine == "arm64":  # Apple Silicon
            download_macos_ffmpeg("arm64")
        else:  # Intel Mac
            download_macos_ffmpeg("x86_64")
    elif system == "Windows":
        download_windows_ffmpeg()
    else:
        print(f"Unsupported system: {system}. Please install FFmpeg manually.")
        return False
    
    # Verify the downloaded binary works
    ffmpeg_path = bin_dir / "ffmpeg"
    if system == "Windows":
        ffmpeg_path = bin_dir / "ffmpeg.exe"
    
    if not ffmpeg_path.exists():
        print(f"ERROR: FFmpeg binary not found at {ffmpeg_path}")
        return False
    
    # Make it executable on Unix-like systems
    if system != "Windows":
        try:
            os.chmod(ffmpeg_path, 0o755)  # rwxr-xr-x
        except Exception as e:
            print(f"ERROR: Could not make FFmpeg executable: {e}")
            return False
    
    # Test the binary
    try:
        result = subprocess.run(
            [str(ffmpeg_path), "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        print(f"FFmpeg version information:\n{result.stdout.splitlines()[0]}")
        print("FFmpeg successfully downloaded and verified!")
        return True
    except subprocess.SubprocessError as e:
        print(f"ERROR: FFmpeg verification failed: {e}")
        return False

def download_macos_ffmpeg(arch):
    """Download and extract FFmpeg for macOS."""
    bin_dir = Path("bin")
    
    # Use a static build of FFmpeg for macOS
    if arch == "arm64":
        url = "https://evermeet.cx/ffmpeg/getrelease/zip"  # Latest static build for arm64
    else:
        url = "https://evermeet.cx/ffmpeg/getrelease/zip"  # Same URL works for both architectures
    
    try:
        # Create a temporary directory for downloading
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            zip_file = tmp_path / "ffmpeg.zip"
            
            # Download the file
            print(f"Downloading FFmpeg from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(zip_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract the zip file
            print("Extracting FFmpeg...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(tmp_path)
            
            # Find the extracted ffmpeg binary
            ffmpeg_binary = None
            for file in tmp_path.glob("*"):
                if file.is_file() and file.name.lower() == "ffmpeg":
                    ffmpeg_binary = file
                    break
            
            if not ffmpeg_binary:
                print("ERROR: Could not find FFmpeg binary in the downloaded package")
                return False
            
            # Copy to bin directory
            shutil.copy2(ffmpeg_binary, bin_dir / "ffmpeg")
            print(f"FFmpeg copied to {bin_dir / 'ffmpeg'}")
            return True
            
    except Exception as e:
        print(f"ERROR: Failed to download and extract FFmpeg: {e}")
        return False

def download_windows_ffmpeg():
    """Download and extract FFmpeg for Windows."""
    bin_dir = Path("bin")
    
    # Use a static build of FFmpeg for Windows
    url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    
    try:
        # Create a temporary directory for downloading
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            zip_file = tmp_path / "ffmpeg.zip"
            
            # Download the file
            print(f"Downloading FFmpeg from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(zip_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract the zip file
            print("Extracting FFmpeg...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(tmp_path)
            
            # Find the extracted ffmpeg binary (it's in a nested directory)
            ffmpeg_binary = None
            for root, dirs, files in os.walk(tmp_path):
                for file in files:
                    if file.lower() == "ffmpeg.exe":
                        ffmpeg_binary = Path(root) / file
                        break
                if ffmpeg_binary:
                    break
            
            if not ffmpeg_binary:
                print("ERROR: Could not find ffmpeg.exe in the downloaded package")
                return False
            
            # Copy to bin directory
            shutil.copy2(ffmpeg_binary, bin_dir / "ffmpeg.exe")
            print(f"FFmpeg copied to {bin_dir / 'ffmpeg.exe'}")
            return True
            
    except Exception as e:
        print(f"ERROR: Failed to download and extract FFmpeg: {e}")
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1) 