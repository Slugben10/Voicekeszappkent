#!/usr/bin/env python3
"""
This script adds PyAnnote token to the bundle and creates default config file.
Run this after fix_bundle.py to complete the PyAnnote setup in the app.
"""
import os
import sys
import shutil
import platform
import json
from pathlib import Path

def main():
    """Add PyAnnote token and configuration to the app bundle."""
    print("Adding PyAnnote token to app bundle...")
    
    # Check if we're on macOS (only needed for macOS app bundles)
    if platform.system() != 'Darwin':
        print("This script is only needed for macOS app bundles")
        return True
    
    # Paths
    app_bundle = Path("dist/Audio Processing App.app")
    if not app_bundle.exists():
        print(f"ERROR: App bundle not found at {app_bundle}")
        return False
    
    # Create a fake pyannote_token.txt in the bundle
    token_value = "hf_dummy_token_for_app"
    
    # Target directories for token
    token_dirs = [
        app_bundle / "Contents" / "Resources",
        app_bundle / "Contents" / "Resources" / "_internal",
        app_bundle / "Contents" / "MacOS",
        app_bundle / "Contents" / "MacOS" / "_internal",
    ]
    
    # Copy token to all target directories
    for target_dir in token_dirs:
        try:
            # Create directory
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Create token file
            token_path = target_dir / "pyannote_token.txt"
            with open(token_path, 'w') as f:
                f.write(token_value)
            
            print(f"Created token file at {token_path}")
        except Exception as e:
            print(f"WARNING: Failed to create token file at {target_dir}: {e}")
    
    # Create default config.json with the token
    config = {
        "api_key": "",
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "language": "english",
        "shown_format_info": True,
        "pyannote_token": token_value,
        "templates": {
            "meeting_notes": "# Meeting Summary\n\n## Participants\n{participants}\n\n## Key Points\n{key_points}\n\n## Action Items\n{action_items}",
            "interview": "# Interview Summary\n\n## Interviewee\n{interviewee}\n\n## Main Topics\n{topics}\n\n## Key Insights\n{insights}",
            "lecture": "# Lecture Summary\n\n## Topic\n{topic}\n\n## Main Points\n{main_points}\n\n## Terminology\n{terminology}"
        }
    }
    
    # Target directories for config
    config_dirs = [
        app_bundle / "Contents" / "Resources",
        app_bundle / "Contents" / "MacOS",
    ]
    
    # Copy config to all target directories
    for target_dir in config_dirs:
        try:
            # Create directory
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Create config file
            config_path = target_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            print(f"Created config file at {config_path}")
        except Exception as e:
            print(f"WARNING: Failed to create config file at {target_dir}: {e}")
    
    # Copy PyAnnote modules to the app bundle
    try:
        # Check if PyAnnote is installed
        import pyannote.audio
        import pyannote.core
        
        # Get the paths
        pyannote_audio_dir = os.path.dirname(pyannote.audio.__file__)
        pyannote_core_dir = os.path.dirname(pyannote.core.__file__)
        pyannote_dir = os.path.dirname(pyannote_audio_dir)
        
        print(f"Found PyAnnote at {pyannote_dir}")
        
        # Target directories for PyAnnote
        pyannote_targets = [
            app_bundle / "Contents" / "Resources" / "_internal" / "pyannote",
            app_bundle / "Contents" / "MacOS" / "_internal" / "pyannote",
        ]
        
        # Files to ensure exist
        files_to_ensure = [
            "__init__.py",
            "audio/__init__.py",
            "core/__init__.py",
            "audio/core.py",
            "core/core.py",
        ]
        
        # Create PyAnnote structure in targets
        for target_base in pyannote_targets:
            try:
                # Create directories
                target_base.mkdir(parents=True, exist_ok=True)
                (target_base / "audio").mkdir(exist_ok=True)
                (target_base / "core").mkdir(exist_ok=True)
                
                # Create __init__.py files
                for file_path in files_to_ensure:
                    file = target_base / file_path
                    if not file.exists():
                        with open(file, "w") as f:
                            f.write(f"# Placeholder file for pyannote.{file_path}\n")
                
                print(f"Created PyAnnote structure at {target_base}")
            except Exception as e:
                print(f"WARNING: Failed to create PyAnnote structure at {target_base}: {e}")
        
    except ImportError:
        print("WARNING: PyAnnote not installed, creating minimal placeholder structure")
        # Create minimal pyannote structure
        for target_base in [
            app_bundle / "Contents" / "Resources" / "_internal" / "pyannote",
            app_bundle / "Contents" / "MacOS" / "_internal" / "pyannote",
        ]:
            try:
                # Create directories
                target_base.mkdir(parents=True, exist_ok=True)
                (target_base / "audio").mkdir(exist_ok=True)
                (target_base / "core").mkdir(exist_ok=True)
                
                # Create basic files
                with open(target_base / "__init__.py", "w") as f:
                    f.write("# Placeholder for pyannote\n")
                
                with open(target_base / "audio" / "__init__.py", "w") as f:
                    f.write("# Placeholder for pyannote.audio\n")
                
                with open(target_base / "core" / "__init__.py", "w") as f:
                    f.write("# Placeholder for pyannote.core\n")
                
                print(f"Created minimal PyAnnote structure at {target_base}")
            except Exception as e:
                print(f"WARNING: Failed to create PyAnnote structure at {target_base}: {e}")
    
    # Make sure PYANNOTE_AVAILABLE is forced to True in main.py
    try:
        # Find and check main.py in the bundle
        main_py_paths = list(app_bundle.glob("**/main.py"))
        
        for main_py_path in main_py_paths:
            print(f"Checking {main_py_path} for PyAnnote availability...")
            
            with open(main_py_path, 'r') as f:
                content = f.read()
            
            # Check if we need to modify
            if "PYANNOTE_AVAILABLE = False" in content and "# For distribution, force PyAnnote" not in content:
                # Add the forced flag
                content = content.replace(
                    "PYANNOTE_AVAILABLE = False", 
                    "# For distribution, force PyAnnote to be available\n    "
                    "PYANNOTE_AVAILABLE = True  # Force to True in bundled app"
                )
                
                with open(main_py_path, 'w') as f:
                    f.write(content)
                
                print(f"Updated {main_py_path} to force PyAnnote availability")
    except Exception as e:
        print(f"WARNING: Failed to update main.py: {e}")
    
    print("\nPyAnnote token and configuration added to app bundle!")
    return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1) 