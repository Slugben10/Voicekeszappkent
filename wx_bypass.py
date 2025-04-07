
# Custom wx.py to bypass screen access check
import os
import sys
import platform

# Set environment variables first
os.environ['PYTHONFRAMEWORK'] = '1'
os.environ['DISPLAY'] = ':0'
os.environ['WX_NO_DISPLAY_CHECK'] = '1'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

# Import the real wx module
import importlib.machinery
import importlib.util

# Store the original import
original_import = __import__

# Define our custom import function
def custom_import(name, *args, **kwargs):
    # For wx._core, we'll patch it
    if name == 'wx._core' or name.startswith('wx._core.'):
        # First get the original module
        module = original_import(name, *args, **kwargs)
        
        # Patch the _macIsRunningOnMainDisplay function if it exists
        if hasattr(module, '_macIsRunningOnMainDisplay'):
            setattr(module, '_macIsRunningOnMainDisplay', lambda: True)
            print("Patched _macIsRunningOnMainDisplay to always return True")
        
        return module
    # For regular imports, use the original import function
    return original_import(name, *args, **kwargs)

# Replace the built-in __import__ function with our custom one
__builtins__['__import__'] = custom_import

# Import the real wx module
import wx as real_wx

# Patch the App class
if hasattr(real_wx, 'App'):
    original_init = real_wx.App.__init__
    
    def patched_init(self, *args, **kwargs):
        kwargs['redirect'] = False
        return original_init(self, *args, **kwargs)
    
    real_wx.App.__init__ = patched_init
    print("Patched wx.App.__init__ to always use redirect=False")

# Make our module act like the real wx module
for attr in dir(real_wx):
    if not attr.startswith('__'):
        globals()[attr] = getattr(real_wx, attr)

# Define the App class with screen check bypassed
class App(real_wx.App):
    def __init__(self, *args, **kwargs):
        kwargs['redirect'] = False
        super().__init__(*args, **kwargs)

# Export everything from the real wx module
__all__ = dir(real_wx)
