"""
PyInstaller runtime hook to set up bundled FFmpeg path.
This will run when the app starts to ensure FFmpeg can be found.
"""
import os
import sys
import platform

def get_bundle_dir():
    """Get the base directory of the application bundle."""
    if getattr(sys, 'frozen', False):
        # Running as a PyInstaller bundle
        if platform.system() == 'Darwin':  # macOS
            # On macOS, the executable is in app_name.app/Contents/MacOS/
            # and the Resources are in app_name.app/Contents/Resources/
            base_dir = os.path.dirname(sys.executable)
            # Print base_dir for debugging
            print(f"App executable path: {base_dir}")
            
            # The correct path should be something like:
            # /path/to/app.app/Contents/Resources
            # But we're seeing:
            # /path/to/dist/Resources (which is wrong)
            
            # Try the correct path first
            correct_resources = os.path.abspath(os.path.join(base_dir, '..', 'Resources'))
            print(f"Looking for correct Resources directory: {correct_resources}")
            
            if os.path.exists(correct_resources):
                print(f"Found correct Resources directory: {correct_resources}")
                return correct_resources
            
            # If that doesn't exist, try to fix the path - the app base dir ends with "/Audio Processing App"
            app_path_parts = base_dir.split(os.sep)
            if app_path_parts and app_path_parts[-1] == "Audio Processing App":
                # We're probably in the wrong path structure, fix it
                correct_app_path = os.path.dirname(os.path.dirname(base_dir))
                corrected_resources = os.path.join(correct_app_path, "Audio Processing App.app", "Contents", "Resources")
                print(f"Trying corrected Resources path: {corrected_resources}")
                if os.path.exists(corrected_resources):
                    print(f"Found corrected Resources directory: {corrected_resources}")
                    return corrected_resources
            
            # If none of that works, try with the pattern from the error logs
            dist_resources = os.path.join(os.path.dirname(os.path.dirname(base_dir)), "Resources")
            print(f"Trying fallback Resources directory: {dist_resources}")
            
            # Finally, just return the resources directory, even if it doesn't exist yet
            # It might be created later
            return dist_resources
        else:
            # On Windows/Linux, the binary is in the root of the app directory
            return os.path.dirname(sys.executable)
    else:
        # Running in development mode
        return os.path.abspath(os.path.dirname(__file__))

def setup_ffmpeg_path():
    """Add the bundled FFmpeg binary to PATH."""
    bundle_dir = get_bundle_dir()
    
    # Construct the path to the bundled FFmpeg binary
    if platform.system() == 'Windows':
        bin_dir = os.path.join(bundle_dir, 'bin')
        ffmpeg_path = os.path.join(bin_dir, 'ffmpeg.exe')
    else:
        bin_dir = os.path.join(bundle_dir, 'bin')
        ffmpeg_path = os.path.join(bin_dir, 'ffmpeg')
    
    # Check if the binary exists
    print(f"Looking for FFmpeg at: {ffmpeg_path}")
    if os.path.exists(ffmpeg_path):
        # Print info for debugging
        print(f"Found bundled FFmpeg at: {ffmpeg_path}")
        
        # Make sure bin_dir is in PATH
        if bin_dir not in os.environ['PATH'].split(os.pathsep):
            os.environ['PATH'] = bin_dir + os.pathsep + os.environ['PATH']
            print(f"Added {bin_dir} to PATH")
        
        # Set a custom environment variable to tell our app where FFmpeg is
        os.environ['BUNDLED_FFMPEG_PATH'] = ffmpeg_path
    else:
        print(f"Warning: Bundled FFmpeg not found at {ffmpeg_path}")
        
        # Try to find it in alternative locations
        alt_locations = [
            # Various locations in the app bundle
            os.path.join(os.path.dirname(bundle_dir), 'MacOS', 'bin', 'ffmpeg'),
            os.path.join(bundle_dir, '..', 'MacOS', 'bin', 'ffmpeg'),
            os.path.join(bundle_dir, '_internal', 'bin', 'ffmpeg'),
            
            # Try to construct paths based on executable path
            os.path.join(os.path.dirname(sys.executable), 'bin', 'ffmpeg'),
            os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'Resources', 'bin', 'ffmpeg'),
            
            # Try some absolute paths that might work
            '/Users/binobenjamin/Documents/Audio/dist/Audio Processing App.app/Contents/Resources/bin/ffmpeg',
            '/Users/binobenjamin/Documents/Audio/dist/Audio Processing App.app/Contents/MacOS/bin/ffmpeg',
        ]
        
        for alt_path in alt_locations:
            print(f"Trying alternative location: {alt_path}")
            if os.path.exists(alt_path):
                print(f"Found FFmpeg at alternative location: {alt_path}")
                alt_bin_dir = os.path.dirname(alt_path)
                os.environ['PATH'] = alt_bin_dir + os.pathsep + os.environ['PATH']
                os.environ['BUNDLED_FFMPEG_PATH'] = alt_path
                break
        else:
            # If we still haven't found FFmpeg, try to install it in the right place
            try:
                # Create bin directory in Resources if it doesn't exist
                os.makedirs(os.path.join(bundle_dir, 'bin'), exist_ok=True)
                
                # Try to copy from a known location
                source_locations = [
                    '/opt/homebrew/bin/ffmpeg',
                    '/usr/local/bin/ffmpeg',
                    '/usr/bin/ffmpeg',
                ]
                
                for source in source_locations:
                    if os.path.exists(source):
                        import shutil
                        target = os.path.join(bundle_dir, 'bin', 'ffmpeg')
                        shutil.copy2(source, target)
                        os.chmod(target, 0o755)  # rwxr-xr-x
                        print(f"Copied FFmpeg from {source} to {target}")
                        
                        # Set environment variables
                        os.environ['PATH'] = os.path.join(bundle_dir, 'bin') + os.pathsep + os.environ['PATH']
                        os.environ['BUNDLED_FFMPEG_PATH'] = target
                        break
            except Exception as e:
                print(f"Failed to install FFmpeg: {e}")

# Run the setup function when this module is imported
setup_ffmpeg_path() 