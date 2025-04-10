#!/usr/bin/env python3
"""
Complete build script to create the app with PyAnnote included.
This runs all necessary steps to produce a standalone app with PyAnnote pre-configured.
"""
import os
import sys
import subprocess
import platform
import time

def run_command(command, description):
    """Run a command and display its output."""
    print(f"\n=== {description} ===")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"ERROR: {description} failed with code {result.returncode}")
        return False
    return True

def main():
    """Run the complete build process."""
    start_time = time.time()
    
    # Step 1: Install PyAnnote
    print("\n=== STEP 1: Installing PyAnnote and dependencies ===")
    if not run_command(f"{sys.executable} install_pyannote.py", "PyAnnote Installation"):
        return False
    
    # Step 2: Build the app
    print("\n=== STEP 2: Building the app ===")
    if not run_command(f"{sys.executable} build_app.py", "App Build"):
        return False
    
    # Step 3: Fix the bundle for FFmpeg
    print("\n=== STEP 3: Fixing app bundle for FFmpeg ===")
    if not run_command(f"{sys.executable} fix_bundle.py", "Bundle Fix"):
        return False
    
    # Step 4: Add PyAnnote configuration to the bundle
    print("\n=== STEP 4: Adding PyAnnote configuration ===")
    if not run_command(f"{sys.executable} fix_pyannote.py", "PyAnnote Configuration"):
        return False
    
    # All done!
    build_time = time.time() - start_time
    print(f"\n=== Build completed in {build_time:.1f} seconds! ===")
    print(f"The app is located in dist/Audio Processing App.app")
    
    # Ask to run the app
    if platform.system() == 'Darwin':
        app_path = "dist/Audio Processing App.app"
        if os.path.exists(app_path):
            print("\nThe app is ready to use.")
            if input("Would you like to run the app now? (y/n): ").lower() == 'y':
                run_command(f"open '{app_path}'", "Opening App")
    
    return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1) 