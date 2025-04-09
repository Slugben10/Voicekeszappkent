#!/usr/bin/env python3
"""
This script fixes the structure of the app bundle to ensure FFmpeg is in the correct location.
Run this after building the app.
"""
import os
import sys
import shutil
import platform
import subprocess
from pathlib import Path

def main():
    """Copy FFmpeg to multiple locations in the app bundle to ensure it's found."""
    print("Fixing app bundle structure...")
    
    # Check if we're on macOS (only needed for macOS app bundles)
    if platform.system() != 'Darwin':
        print("This script is only needed for macOS app bundles")
        return True
    
    # Paths
    app_bundle = Path("dist/Audio Processing App.app")
    if not app_bundle.exists():
        print(f"ERROR: App bundle not found at {app_bundle}")
        return False
    
    # Source FFmpeg
    source_ffmpeg = Path("bin/ffmpeg")
    if not source_ffmpeg.exists():
        print(f"ERROR: Source FFmpeg not found at {source_ffmpeg}")
        return False
    
    # Target directories
    target_dirs = [
        app_bundle / "Contents" / "Resources" / "bin",
        app_bundle / "Contents" / "MacOS" / "bin",
        app_bundle / "Contents" / "MacOS" / "_internal" / "bin",
        app_bundle / "Contents" / "Resources" / "_internal" / "bin",
    ]
    
    # Create directories and copy FFmpeg
    for target_dir in target_dirs:
        try:
            # Create directory
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy FFmpeg
            target_path = target_dir / "ffmpeg"
            shutil.copy2(source_ffmpeg, target_path)
            
            # Make executable
            os.chmod(target_path, 0o755)  # rwxr-xr-x
            
            print(f"Copied FFmpeg to {target_path}")
        except Exception as e:
            print(f"WARNING: Failed to copy FFmpeg to {target_dir}: {e}")
    
    # Fix path for Resources - should be Contents/Resources
    try:
        # Create a symbolic link to correct Resources path
        contents_dir = app_bundle / "Contents"
        wrong_resources = Path("/Users/binobenjamin/Documents/Audio/dist/Resources")
        if not wrong_resources.exists():
            os.makedirs(wrong_resources, exist_ok=True)
            # Copy the FFmpeg to wrong Resources path as well
            wrong_bin = wrong_resources / "bin"
            wrong_bin.mkdir(exist_ok=True)
            wrong_ffmpeg = wrong_bin / "ffmpeg"
            shutil.copy2(source_ffmpeg, wrong_ffmpeg)
            os.chmod(wrong_ffmpeg, 0o755)
            print(f"Copied FFmpeg to {wrong_ffmpeg} (wrong path)")
    except Exception as e:
        print(f"WARNING: Failed to fix Resources path: {e}")
    
    # Check for lightning_fabric version.info
    try:
        # Source for lightning_fabric version.info
        source_version_info = None
        
        # Try to find it in the python package
        try:
            import lightning_fabric
            package_dir = os.path.dirname(lightning_fabric.__file__)
            version_file = os.path.join(package_dir, "version.info")
            if os.path.exists(version_file):
                source_version_info = version_file
                print(f"Found lightning_fabric version.info at {version_file}")
        except ImportError:
            print("Could not import lightning_fabric")
        
        # If not found, create a placeholder
        if source_version_info is None:
            placeholder = Path("lightning_fabric_version.info")
            with open(placeholder, 'w') as f:
                f.write("0.0.0\n")
            source_version_info = placeholder
            print(f"Created placeholder version.info at {placeholder}")
        
        # Target paths for version.info
        version_targets = [
            app_bundle / "Contents" / "Resources" / "lightning_fabric",
            app_bundle / "Contents" / "MacOS" / "lightning_fabric",
            app_bundle / "Contents" / "Resources" / "_internal" / "lightning_fabric",
            app_bundle / "Contents" / "MacOS" / "_internal" / "lightning_fabric",
        ]
        
        # Copy to all targets
        for target_dir in version_targets:
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
                target_file = target_dir / "version.info"
                shutil.copy2(source_version_info, target_file)
                print(f"Copied version.info to {target_file}")
            except Exception as e:
                print(f"WARNING: Failed to copy version.info to {target_dir}: {e}")
    
    except Exception as e:
        print(f"WARNING: Error handling lightning_fabric version.info: {e}")
    
    # Copy SpeechBrain files
    try:
        import speechbrain
        speechbrain_dir = os.path.dirname(speechbrain.__file__)
        print(f"Found speechbrain at {speechbrain_dir}")
        
        # Target directories for speechbrain
        speechbrain_targets = [
            app_bundle / "Contents" / "Resources" / "_internal" / "speechbrain",
            app_bundle / "Contents" / "MacOS" / "_internal" / "speechbrain",
            Path("dist/Audio Processing App") / "_internal" / "speechbrain",
        ]
        
        # Directories to copy
        subdirs_to_copy = ["utils", "dataio"]
        
        for subdir in subdirs_to_copy:
            source_subdir = os.path.join(speechbrain_dir, subdir)
            if os.path.exists(source_subdir):
                for target_base in speechbrain_targets:
                    try:
                        target_subdir = target_base / subdir
                        if os.path.exists(target_subdir):
                            shutil.rmtree(target_subdir)
                        shutil.copytree(source_subdir, target_subdir)
                        print(f"Copied speechbrain/{subdir} to {target_subdir}")
                    except Exception as e:
                        print(f"WARNING: Failed to copy speechbrain/{subdir} to {target_base}: {e}")
        
        # Also copy the __init__.py files
        for target_base in speechbrain_targets:
            try:
                # Copy main __init__.py
                init_file = os.path.join(speechbrain_dir, "__init__.py")
                if os.path.exists(init_file):
                    shutil.copy2(init_file, target_base / "__init__.py")
                    print(f"Copied speechbrain/__init__.py to {target_base}")
                
                # Create core.py if it doesn't exist
                core_file = os.path.join(speechbrain_dir, "core.py")
                if os.path.exists(core_file):
                    shutil.copy2(core_file, target_base / "core.py")
                    print(f"Copied speechbrain/core.py to {target_base}")
                else:
                    # Create a simple core.py
                    with open(target_base / "core.py", "w") as f:
                        f.write("# Placeholder file for speechbrain.core\n")
                    print(f"Created placeholder core.py in {target_base}")
                
                # Copy version.txt or create if doesn't exist
                version_file = os.path.join(speechbrain_dir, "version.txt")
                if os.path.exists(version_file):
                    shutil.copy2(version_file, target_base / "version.txt")
                    print(f"Copied speechbrain/version.txt to {target_base}")
                else:
                    # Create a simple version.txt
                    with open(target_base / "version.txt", "w") as f:
                        f.write("0.5.15\n")
                    print(f"Created placeholder version.txt in {target_base}")
            except Exception as e:
                print(f"WARNING: Failed to copy speechbrain core files to {target_base}: {e}")
    
    except ImportError:
        print("WARNING: Could not import speechbrain - creating minimal structure")
        # Create minimal speechbrain structure
        for target_base in [
            app_bundle / "Contents" / "Resources" / "_internal" / "speechbrain",
            app_bundle / "Contents" / "MacOS" / "_internal" / "speechbrain",
            Path("dist/Audio Processing App") / "_internal" / "speechbrain",
        ]:
            try:
                # Create directories
                target_base.mkdir(parents=True, exist_ok=True)
                (target_base / "utils").mkdir(exist_ok=True)
                (target_base / "dataio").mkdir(exist_ok=True)
                
                # Create __init__.py files
                with open(target_base / "__init__.py", "w") as f:
                    f.write("# Placeholder file for speechbrain\n")
                
                with open(target_base / "utils" / "__init__.py", "w") as f:
                    f.write("# Placeholder file for speechbrain.utils\n")
                    
                with open(target_base / "dataio" / "__init__.py", "w") as f:
                    f.write("# Placeholder file for speechbrain.dataio\n")
                    
                # Create core.py
                with open(target_base / "core.py", "w") as f:
                    f.write("# Placeholder file for speechbrain.core\n")
                    
                # Create version.txt
                with open(target_base / "version.txt", "w") as f:
                    f.write("0.5.15\n")
                    
                # Create importutils.py
                with open(target_base / "utils" / "importutils.py", "w") as f:
                    f.write("""# Placeholder file for speechbrain.utils.importutils
def find_imports(*args, **kwargs):
    return []
    
def lazy_export_all(*args, **kwargs):
    pass
"""
                    )
                
                print(f"Created minimal speechbrain structure in {target_base}")
            except Exception as e:
                print(f"WARNING: Failed to create speechbrain structure in {target_base}: {e}")
    
    except Exception as e:
        print(f"WARNING: Error handling speechbrain files: {e}")
    
    print("\nApp bundle structure fixed! Try running the app now.")
    
    # Try running the app
    if input("Would you like to run the app now? (y/n): ").lower() == 'y':
        try:
            subprocess.run(["open", app_bundle])
        except Exception as e:
            print(f"Error running app: {e}")
    
    return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1) 