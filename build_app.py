#!/usr/bin/env python3
import os
import sys
import shutil
import platform
import subprocess
import importlib.util
from pathlib import Path
import inspect

# Bypass the screen access check directly in this script
if platform.system() == "Darwin":  # macOS
    os.environ['PYTHONFRAMEWORK'] = '1'
    os.environ['DISPLAY'] = ':0'
    os.environ['WX_NO_DISPLAY_CHECK'] = '1'
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

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
    
    # Remove runtime hook file if it exists
    hook_path = Path("disable_screen_check.py")
    if hook_path.exists():
        os.remove(hook_path)

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

def create_runtime_hook():
    """Create a runtime hook to disable screen access check."""
    hook_content = """
# PyInstaller runtime hook to disable screen access check
import os
import sys
import platform

def disable_screen_check():
    \"\"\"Disable the macOS screen access check.\"\"\"
    if platform.system() == 'Darwin':
        # Set environment variables to bypass framework check
        os.environ['PYTHONFRAMEWORK'] = '1'
        os.environ['DISPLAY'] = ':0'
        os.environ['WX_NO_DISPLAY_CHECK'] = '1'
        os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
        
        try:
            # Try to patch the sys.meta_path to use our custom wx module
            import wx_bypass
            sys.modules['wx'] = wx_bypass
            print("Successfully replaced wx with custom implementation")
            
            # Try to patch wx.App to disable the check at runtime
            import wx
            if hasattr(wx, 'App'):
                original_init = wx.App.__init__
                
                def patched_init(self, *args, **kwargs):
                    # Make sure redirect is False
                    kwargs['redirect'] = False
                    return original_init(self, *args, **kwargs)
                
                wx.App.__init__ = patched_init
                print("Successfully patched wx.App.__init__")
                
                # Also try to patch the _core check function if it exists
                if hasattr(wx, '_core'):
                    core = wx._core
                    if hasattr(core, '_macIsRunningOnMainDisplay'):
                        # Replace with a function that always returns True
                        core._macIsRunningOnMainDisplay = lambda: True
                        print("Successfully patched wx._core._macIsRunningOnMainDisplay")
        except ImportError as e:
            print(f"ImportError: {e}")
            pass
        except Exception as e:
            print(f"Error patching wx: {e}")

# Call the function immediately when this hook is loaded
disable_screen_check()
"""
    
    hook_path = "disable_screen_check.py"
    with open(hook_path, "w") as f:
        f.write(hook_content)
    
    return hook_path

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

def create_wx_bypass():
    """Create a custom wx.py file that bypasses the screen check."""
    wx_bypass_content = """
# Custom wx.py to bypass screen access check
import os
import sys
import platform
import importlib.util
import importlib.machinery

# Set environment variables first
os.environ['PYTHONFRAMEWORK'] = '1'
os.environ['DISPLAY'] = ':0'
os.environ['WX_NO_DISPLAY_CHECK'] = '1'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

# First, load the real wx module so we can access it
try:
    # Get the real wx module path
    real_wx_spec = importlib.util.find_spec('wx')
    if not real_wx_spec:
        raise ImportError("Can't find the real wx module")
    
    # Import the real wx module
    real_wx = importlib.util.module_from_spec(real_wx_spec)
    real_wx_spec.loader.exec_module(real_wx)
    
    # Patch the _macIsRunningOnMainDisplay function if it exists in _core
    if hasattr(real_wx, '_core'):
        if hasattr(real_wx._core, '_macIsRunningOnMainDisplay'):
            setattr(real_wx._core, '_macIsRunningOnMainDisplay', lambda: True)
            print("Patched _macIsRunningOnMainDisplay to always return True")
    
    # Patch the App class
    if hasattr(real_wx, 'App'):
        original_init = real_wx.App.__init__
        
        def patched_init(self, *args, **kwargs):
            kwargs['redirect'] = False
            return original_init(self, *args, **kwargs)
        
        real_wx.App.__init__ = patched_init
        print("Patched wx.App.__init__ to always use redirect=False")
    
    # Copy all attributes from the real wx module to this module
    for attr in dir(real_wx):
        if not attr.startswith('__'):
            globals()[attr] = getattr(real_wx, attr)
    
    # Provide an import hook for submodules
    class WxFinder:
        def find_spec(self, fullname, path=None, target=None):
            # Only handle wx.* imports
            if not fullname.startswith('wx.'):
                return None
            
            # Get the real submodule name (e.g., 'adv' from 'wx.adv')
            submodule = fullname.split('.')[-1]
            parent_name = '.'.join(fullname.split('.')[:-1])
            
            # Try to find the real submodule
            try:
                parent = sys.modules.get(parent_name)
                if not parent:
                    return None
                
                # Find the real submodule in the original wx package
                if hasattr(real_wx, submodule):
                    # Return the real submodule
                    return importlib.util.find_spec(f'wx.{submodule}')
            except (ImportError, AttributeError):
                return None
    
    # Register our finder in sys.meta_path
    sys.meta_path.insert(0, WxFinder())
    
    # Define any special handling needed
    class App(real_wx.App):
        def __init__(self, *args, **kwargs):
            kwargs['redirect'] = False
            real_wx.App.__init__(self, *args, **kwargs)
    
    # Export everything from the real wx module
    __all__ = dir(real_wx)

except ImportError as e:
    print(f"Error importing real wx module: {e}")
    # Fallback to a minimal implementation
    class App:
        def __init__(self, *args, **kwargs):
            print("Using fallback App class")
        def MainLoop(self):
            print("Fallback MainLoop called")
    
    # Define bare minimum classes to prevent crashes
    class Frame:
        def __init__(self, *args, **kwargs):
            pass
    class Panel:
        def __init__(self, *args, **kwargs):
            pass
    class BoxSizer:
        def __init__(self, *args, **kwargs):
            pass
    class ID_ANY:
        pass
    VERTICAL = 0
    HORIZONTAL = 1
"""
    
    # Write the file
    wx_bypass_path = "wx_bypass.py"
    with open(wx_bypass_path, "w") as f:
        f.write(wx_bypass_content)
    
    return wx_bypass_path

def simplify_build():
    """Build the application using the simplest possible approach."""
    print("Building application using simplified approach...")
    
    # Create runtime hook
    hook_path = create_runtime_hook()
    
    # Create wx bypass
    wx_bypass_path = create_wx_bypass()
    
    # Find required data files
    additional_data = find_package_data_files()
    
    # Create PyInstaller command with the most basic options
    cmd = [
        "pyinstaller",
        "--name", APP_NAME,
        "--onedir",
        "--windowed",
        "--clean",
        "--runtime-hook", hook_path,
        # Add the wx bypass module as additional data
        "--add-data", f"{wx_bypass_path}:.",
        # Exclude problematic torch libraries
        "--exclude-module", "torch",
        "--exclude-module", "torchvision",
        "--exclude-module", "torchaudio",
    ]
    
    # Add icon if exists
    icon_path = get_icon_path()
    if icon_path and os.path.exists(icon_path):
        cmd.extend(["--icon", icon_path])
    
    # Add hidden imports
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
    result = subprocess.run(cmd)
    return result.returncode == 0

def alternative_build():
    """Try an alternative build approach if the first approach fails."""
    print("Trying alternative build approach...")
    
    # Create runtime hook
    hook_path = create_runtime_hook()
    
    # Create wx bypass
    wx_bypass_path = create_wx_bypass()
    
    # Find required data files
    additional_data = find_package_data_files()
    
    # Create PyInstaller command with alternative options
    cmd = [
        "pyinstaller",
        "--name", APP_NAME,
        "--onedir",
        "--windowed",
        "--clean",
        "--noupx",  # Disable UPX compression
        "--runtime-hook", hook_path,
        # Add the wx bypass module as additional data
        "--add-data", f"{wx_bypass_path}:.",
    ]
    
    # Add icon if exists
    icon_path = get_icon_path()
    if icon_path and os.path.exists(icon_path):
        cmd.extend(["--icon", icon_path])
    
    # Add hidden imports
    cmd.extend([
        "--hidden-import", "openai",
        "--hidden-import", "wx",
        "--hidden-import", "wx.adv",
        "--collect-submodules", "wx",
    ])
    
    # Add data files
    for data_spec in additional_data:
        cmd.extend(["--add-data", data_spec])
    
    # Add main script
    cmd.append("main.py")
    
    # Run PyInstaller
    result = subprocess.run(cmd)
    return result.returncode == 0

def spec_file_build():
    """Create a custom spec file that excludes problematic libraries."""
    print("Trying spec file build approach...")
    
    # Create runtime hook
    hook_path = create_runtime_hook()
    
    # Create wx bypass
    wx_bypass_path = create_wx_bypass()
    
    # Create a custom spec file
    spec_content = f"""
# -*- mode: python ; coding: utf-8 -*-
import sys
import os

# Define what modules to exclude
excluded_modules = ['torch', 'torchvision', 'torchaudio', 'tensorflow']

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('{wx_bypass_path}', '.')],
    hiddenimports=['openai', 'wx', 'wx.adv'],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=['{hook_path}'],
    excludes=excluded_modules,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# Filter out any excluded module binaries/data
a.binaries = [x for x in a.binaries if not any(excluded in x[0] for excluded in excluded_modules)]
a.datas = [x for x in a.datas if not any(excluded in x[0] for excluded in excluded_modules)]

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='{APP_NAME}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='{APP_NAME}',
)

# Only add BUNDLE for macOS
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='{APP_NAME}.app',
        icon='icon.icns' if os.path.exists('icon.icns') else None,
        bundle_identifier=None,
        info_plist={{
            'NSPrincipalClass': 'NSApplication',
            'NSAppleScriptEnabled': False,
            'LSEnvironment': {{
                'PYTHONFRAMEWORK': '1',
                'DISPLAY': ':0',
                'WX_NO_DISPLAY_CHECK': '1',
                'OBJC_DISABLE_INITIALIZE_FORK_SAFETY': 'YES',
            }},
        }},
    )
"""
    
    # Write the spec file
    spec_path = f"{APP_NAME}.spec"
    with open(spec_path, "w") as f:
        f.write(spec_content)
    
    # Run PyInstaller with the spec file
    result = subprocess.run(["pyinstaller", spec_path])
    return result.returncode == 0

def find_and_patch_wx_sources():
    """Find and directly patch wxPython source code to disable screen check."""
    print("Searching for wxPython source files to patch...")
    
    # First try to find the Python executable
    python_exe = sys.executable
    
    # Run a small script to find the wxPython package location
    find_wx_script = """
import sys
try:
    import wx
    print(wx.__file__)
except ImportError:
    print("wx not found")
except Exception as e:
    print(f"Error: {e}")
"""
    result = subprocess.run([python_exe, "-c", find_wx_script], capture_output=True, text=True)
    wx_init_path = result.stdout.strip()
    
    if not wx_init_path or "not found" in wx_init_path or "Error" in wx_init_path:
        print(f"Could not find wxPython: {wx_init_path}")
        return False
    
    # Get the base wxPython directory
    wx_dir = os.path.dirname(wx_init_path)
    
    # Look for _core.py or similar files
    potential_files = [
        os.path.join(wx_dir, "_core.py"),
        os.path.join(wx_dir, "core.py"),
        os.path.join(wx_dir, "_core", "__init__.py"),
        os.path.join(wx_dir, "core", "__init__.py"),
        os.path.join(wx_dir, "lib", "wxp", "wxPython_core.py"),
    ]
    
    # Also look for .so or .dylib files
    for root, dirs, files in os.walk(wx_dir):
        for file in files:
            if "_core" in file and (file.endswith(".so") or file.endswith(".dylib")):
                potential_files.append(os.path.join(root, file))
    
    # Flag to track if we found and patched any files
    patched_any = False
    
    # Check each potential file
    for filepath in potential_files:
        if os.path.exists(filepath):
            file_ext = os.path.splitext(filepath)[1]
            
            # Different approach for binary (.so/.dylib) vs. Python (.py) files
            if file_ext in ['.so', '.dylib']:
                # Binary file - create a wrapper
                try:
                    dirname = os.path.dirname(filepath)
                    filename = os.path.basename(filepath)
                    
                    # Create a wrapper Python file in the same directory
                    wrapper_path = os.path.join(dirname, "_core_wrapper.py")
                    wrapper_content = f"""
# Wrapper for {filename} to disable screen check
import os
import sys
import importlib.util
import importlib.machinery

# Set environment variables
os.environ['PYTHONFRAMEWORK'] = '1'
os.environ['DISPLAY'] = ':0'
os.environ['WX_NO_DISPLAY_CHECK'] = '1'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

# Load the original module
original_path = "{filepath}"
loader = importlib.machinery.ExtensionFileLoader('_core', original_path)
original_module = loader.load_module()

# Replace the screen check function
if hasattr(original_module, '_macIsRunningOnMainDisplay'):
    original_module._macIsRunningOnMainDisplay = lambda: True
    print("Patched _macIsRunningOnMainDisplay in binary module")

# Copy all attributes to this module
for attr in dir(original_module):
    if not attr.startswith('__'):
        globals()[attr] = getattr(original_module, attr)

# Explicitly add a correct version of the function
def _macIsRunningOnMainDisplay():
    return True

# Export everything
__all__ = dir(original_module)
"""
                    with open(wrapper_path, 'w') as f:
                        f.write(wrapper_content)
                    
                    # Create an empty __init__.py if needed
                    init_path = os.path.join(dirname, "__init__.py")
                    if not os.path.exists(init_path):
                        with open(init_path, 'w') as f:
                            f.write("# Created by build script\n")
                    
                    print(f"Created wrapper for {filepath}")
                    patched_any = True
                except Exception as e:
                    print(f"Error creating wrapper for {filepath}: {e}")
            
            elif file_ext == '.py':
                try:
                    # Backup the file
                    backup_path = filepath + ".bak"
                    if not os.path.exists(backup_path):
                        shutil.copy(filepath, backup_path)
                    
                    # Read the file
                    with open(filepath, 'r') as f:
                        content = f.read()
                    
                    # Look for the screen check function
                    if "macIsRunningOnMainDisplay" in content:
                        # Replace the function with one that always returns True
                        if "def _macIsRunningOnMainDisplay():" in content:
                            patched_content = content.replace(
                                "def _macIsRunningOnMainDisplay():", 
                                "def _macIsRunningOnMainDisplay():\n    return True  # Patched by build script"
                            )
                            
                            # Also remove any logic inside the function
                            import re
                            patched_content = re.sub(
                                r'def _macIsRunningOnMainDisplay\(\):\s+[^#].*?return',
                                'def _macIsRunningOnMainDisplay():\n    return True  # Patched by build script\n    #return',
                                patched_content,
                                flags=re.DOTALL
                            )
                            
                            with open(filepath, 'w') as f:
                                f.write(patched_content)
                            
                            print(f"Patched screen check function in {filepath}")
                            patched_any = True
                    
                    # If there's App.__init__ with redirect logic, patch that too
                    if "App" in content and "__init__" in content and "redirect" in content:
                        with open(filepath, 'r') as f:
                            content = f.read()
                        
                        # Find the App class and its __init__ method
                        import re
                        match = re.search(r'class\s+App.*?def\s+__init__\s*\([^)]*\)\s*:', content, re.DOTALL)
                        if match:
                            init_pos = match.end()
                            # Add code to force redirect=False
                            patched_content = content[:init_pos] + "\n        # Force redirect=False - patched by build script\n        kwargs['redirect'] = False\n" + content[init_pos:]
                            
                            with open(filepath, 'w') as f:
                                f.write(patched_content)
                            
                            print(f"Patched App.__init__ in {filepath}")
                            patched_any = True
                except Exception as e:
                    print(f"Error patching {filepath}: {e}")
    
    return patched_any

def main():
    """Main function to build the application."""
    # Clean output directories
    clean_output()
    
    # Try the most direct approach - patch wxPython source files
    patched_wx_sources = find_and_patch_wx_sources()
    if patched_wx_sources:
        print("Successfully patched wxPython source files to disable screen check")
    
    # For the direct approach, add wx.pth file to redirect _core import
    wx_path_content = """
# Patch _core to bypass screen access check
import sys, os
import importlib.util

# Try to modify the import path for wx._core if needed
try:
    import wx
    wx_dir = os.path.dirname(wx.__file__)
    # Check if our wrapper exists
    wrapper_path = os.path.join(wx_dir, "_core_wrapper.py")
    if os.path.exists(wrapper_path):
        # Override _core with our wrapper
        spec = importlib.util.spec_from_file_location("wx._core", wrapper_path)
        _core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_core)
        sys.modules['wx._core'] = _core
        print("Redirected wx._core to wrapper")
except Exception as e:
    print(f"Error setting up wx._core wrapper: {e}")
"""
    wx_path_file = "wx_path.py"
    with open(wx_path_file, "w") as f:
        f.write(wx_path_content)
    
    # Simple manual patch to main.py to disable screen check
    if platform.system() == "Darwin" and os.path.exists("main.py"):
        print("Applying comprehensive patch to main.py...")
        
        # Backup original file
        shutil.copy("main.py", "main.py.bak")
        
        with open("main.py", "r") as f:
            content = f.read()
        
        # Add our comprehensive patch at the very beginning of the file
        patch = """# COMPREHENSIVE SCREEN CHECK BYPASS - AUTO-PATCHED
import os
import sys
import platform

# For wxPython on macOS, bypass framework/screen checks
os.environ['PYTHONFRAMEWORK'] = '1'
os.environ['DISPLAY'] = ':0'
os.environ['WX_NO_DISPLAY_CHECK'] = '1'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

# Simple function that just returns True for screen check
def always_return_true(*args, **kwargs):
    return True

# Direct approach - import wx and replace classes/functions
try:
    # First try direct patching of _macIsRunningOnMainDisplay
    import wx
    
    if hasattr(wx, '_core'):
        if hasattr(wx._core, '_macIsRunningOnMainDisplay'):
            wx._core._macIsRunningOnMainDisplay = always_return_true
            print("Patched wx._core._macIsRunningOnMainDisplay")
    
    # Completely replace App class instead of patching its __init__
    if hasattr(wx, 'App'):
        # Store reference to original App class
        OriginalApp = wx.App
        
        # Create our safe App class
        class SafeApp(OriginalApp):
            def __init__(self, *args, **kwargs):
                # Extract all arguments with safe defaults
                redirect = False  # Always force redirect to False
                filename = None
                useBestVisual = False
                clearSigInt = True
                
                # Extract values from kwargs if provided
                if 'filename' in kwargs:
                    filename = kwargs['filename']
                if 'useBestVisual' in kwargs:
                    useBestVisual = kwargs['useBestVisual']
                if 'clearSigInt' in kwargs:
                    clearSigInt = kwargs['clearSigInt']
                
                # Override with positional args if provided
                if len(args) > 1:
                    filename = args[1]
                if len(args) > 2:
                    useBestVisual = args[2]
                if len(args) > 3:
                    clearSigInt = args[3]
                
                # Call the PyApp.__init__ directly without using kwargs
                if hasattr(wx, 'PyApp'):
                    wx.PyApp.__init__(self)
                else:
                    # Use super() on the parent class of App
                    super(OriginalApp, self).__init__()
                
                # Replicate the code from original App.__init__ but skip the screen check
                if useBestVisual:
                    self.SetUseBestVisual(useBestVisual)
                
                if clearSigInt:
                    try:
                        import signal
                        signal.signal(signal.SIGINT, signal.SIG_DFL)
                    except:
                        pass
                
                # Handle redirect but force it to False
                self.stdioWin = None
                if hasattr(sys, 'stdout') and hasattr(sys, 'stderr'):
                    self.saveStdio = (sys.stdout, sys.stderr)
                else:
                    self.saveStdio = (None, None)
                
                # Important: Don't actually redirect
                # if redirect:
                #     self.RedirectStdio(filename)
                
                # Set install prefix
                if hasattr(wx, 'StandardPaths'):
                    prefix = sys.prefix
                    if isinstance(prefix, (bytes, bytearray)):
                        prefix = prefix.decode(sys.getfilesystemencoding())
                    wx.StandardPaths.Get().SetInstallPrefix(prefix)
                
                # Set Mac-specific options if available
                if hasattr(wx, 'SystemOptions'):
                    wx.SystemOptions.SetOption("mac.listctrl.always_use_generic", 1)
                
                # Finalize bootstrap
                if hasattr(self, '_BootstrapApp'):
                    self._BootstrapApp()
        
        # Replace the App class
        wx.App = SafeApp
        print("Replaced wx.App with completely reimplemented SafeApp")
    
    # Also patch individual modules
    for module_name in list(sys.modules.keys()):
        if module_name.startswith('wx.'):
            module = sys.modules[module_name]
            # Patch screen check function if it exists
            if hasattr(module, '_macIsRunningOnMainDisplay'):
                module._macIsRunningOnMainDisplay = always_return_true
                print(f"Patched _macIsRunningOnMainDisplay in {module_name}")
except Exception as e:
    print(f"wx patching error (non-fatal): {e}")

# Force import core modules early
try:
    import wx.adv
    import wx.core
    print("Successfully pre-loaded wx modules")
except Exception as e:
    print(f"Error pre-loading wx modules (non-fatal): {e}")
"""
        
        # Write patched file
        with open("main.py", "w") as f:
            f.write(patch + content)
        
        print("main.py successfully patched with comprehensive bypass")
    
    # Try direct build approach
    print("\n--- FINAL APPROACH: Comprehensive patched build ---")
    cmd = [
        "pyinstaller",
        "--name", APP_NAME,
        "--onedir",
        "--windowed",
        "--clean",
        # Add our path file as a runtime hook
        "--runtime-hook", wx_path_file,
        # Add data files
        "--add-data", f"{wx_path_file}:.",
        # Exclude problematic libraries
        "--exclude-module", "torch",
        "--exclude-module", "torchvision",
        "--exclude-module", "torchaudio",
    ]
    
    # Add icon if exists
    icon_path = get_icon_path()
    if icon_path and os.path.exists(icon_path):
        cmd.extend(["--icon", icon_path])
    
    # Add hidden imports
    cmd.extend([
        "--hidden-import", "openai",
        "--hidden-import", "wx",
        "--hidden-import", "wx.adv",
    ])
    
    # Add main script
    cmd.append("main.py")
    
    # Run PyInstaller
    result = subprocess.run(cmd)
    success = result.returncode == 0
    
    # Restore original main.py
    if os.path.exists("main.py.bak"):
        shutil.move("main.py.bak", "main.py")
        print("Restored original main.py")
    
    # Restore any patched wx files
    for file_path in Path(".").glob("**/*.bak"):
        try:
            original_path = str(file_path)[:-4]  # Remove .bak
            shutil.move(file_path, original_path)
            print(f"Restored {original_path}")
        except Exception as e:
            print(f"Error restoring {file_path}: {e}")
    
    # Delete the temporary path file
    if os.path.exists(wx_path_file):
        os.remove(wx_path_file)
    
    # Create additional directories
    create_directories()
    
    if success:
        print(f"Build completed. App is located in {OUTPUT_DIR}/{APP_NAME}.app")
        return 0
    else:
        print("\nComprehensive build approach failed.")
        print("Try patching wxPython source files directly:")
        print("1. Find the wxPython installation (run 'python -c \"import wx; print(wx.__file__)\"')")
        print("2. Locate _core.py or similar files")
        print("3. Replace the _macIsRunningOnMainDisplay function with one that just returns True")
        print("4. Then run: pyinstaller --name 'Audio Processing App' --onedir --windowed main.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())