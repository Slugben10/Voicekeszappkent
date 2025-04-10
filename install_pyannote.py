#!/usr/bin/env python3
"""
Script to install PyAnnote and bundle it with the application.
This will install the necessary PyAnnote modules and create a default token.
"""

import os
import sys
import subprocess
import platform
import argparse

def install_pyannote():
    """Install PyAnnote and its dependencies."""
    print("Installing PyAnnote and dependencies...")
    
    # First check if we're in the virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("WARNING: Not running in a virtual environment. Recommend using a venv to avoid conflicts.")
        
        # Ask for confirmation
        response = input("Continue with installation outside of virtual environment? (y/n): ")
        if response.lower() != 'y':
            print("Installation aborted.")
            return False
    
    # Install dependencies
    try:
        # Install PyTorch and torchaudio first
        print("Installing PyTorch and torchaudio...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchaudio"
        ])
        
        # Install PyAnnote
        print("Installing PyAnnote...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "pyannote.audio"
        ])
        
        print("PyAnnote installation completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing PyAnnote: {e}")
        return False

def create_mock_token(token_value="hf_mock_token"):
    """Create a mock token for PyAnnote to prevent prompts."""
    print("Creating mock PyAnnote token...")
    
    # Determine the location to store the token
    if platform.system() == 'darwin' and getattr(sys, 'frozen', False):
        # For macOS app bundle, use proper location
        home_dir = os.path.expanduser("~")
        config_dir = os.path.join(home_dir, "Documents", "Audio Processing App")
        os.makedirs(config_dir, exist_ok=True)
        token_file = os.path.join(config_dir, "pyannote_token.txt")
    else:
        # For development environment, use current directory
        token_file = "pyannote_token.txt"
    
    try:
        # Write token to file
        with open(token_file, 'w') as f:
            f.write(token_value)
        print(f"Mock token created at: {token_file}")
        return True
    except Exception as e:
        print(f"Error creating mock token: {e}")
        return False

def update_config(token_value="hf_mock_token"):
    """Update the config.json file with the token."""
    print("Updating config.json with PyAnnote token...")
    
    # Determine config file location
    if platform.system() == 'darwin' and getattr(sys, 'frozen', False):
        # For macOS app bundle, use proper location
        home_dir = os.path.expanduser("~")
        config_dir = os.path.join(home_dir, "Documents", "Audio Processing App")
        config_file = os.path.join(config_dir, "config.json")
    else:
        # For development environment, use current directory
        config_file = "config.json"
    
    import json
    
    # Read existing config or create new one
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError:
            # If file exists but is corrupted, start with a new config
            config = {}
    else:
        config = {}
    
    # Update token
    config["pyannote_token"] = token_value
    
    try:
        # Save updated config
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Config updated at: {config_file}")
        return True
    except Exception as e:
        print(f"Error updating config: {e}")
        return False

def main():
    """Main function to install and configure PyAnnote."""
    parser = argparse.ArgumentParser(description="Install and configure PyAnnote for bundling")
    parser.add_argument("--token", default="hf_dummy_token", help="Default token to use")
    args = parser.parse_args()
    
    print("=== PyAnnote Installer ===")
    
    # Install PyAnnote
    if install_pyannote():
        # Create mock token
        create_mock_token(args.token)
        
        # Update config
        update_config(args.token)
        
        print("\nPyAnnote installation and configuration completed.")
        print("Now you can build the application with PyAnnote included.")
    else:
        print("\nPyAnnote installation failed.")
        sys.exit(1)

if __name__ == "__main__":
    main() 