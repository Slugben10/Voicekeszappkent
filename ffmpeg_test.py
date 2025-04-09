import os
import subprocess
import sys

def check_ffmpeg():
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"PATH environment variable: {os.environ.get('PATH', 'PATH not found')}")
    
    # Try both with full path and without
    ffmpeg_paths = [
        "ffmpeg",
        "/opt/homebrew/bin/ffmpeg"
    ]
    
    for cmd in ffmpeg_paths:
        print(f"\nTrying to run: {cmd} -version")
        try:
            result = subprocess.run(
                [cmd, "-version"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=True
            )
            print(f"Success! Output: {result.stdout[:100].decode() if result.stdout else 'No output'}")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"Failed with error: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    check_ffmpeg() 