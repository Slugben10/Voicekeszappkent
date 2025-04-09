
# Simple monkey patch for wx._core._macIsRunningOnMainDisplay
# This file gets imported before the real wx module

import sys
import os

# Tell wxPython to not check for the framework
os.environ['PYTHONFRAMEWORK'] = '1'
os.environ['DISPLAY'] = ':0'
os.environ['WX_NO_DISPLAY_CHECK'] = '1'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

# Define a fake function that always returns True
def _fake_mac_display_check(*args, **kwargs):
    return True

# Store the original import
_original_import = __import__

# Create a patched import function
def _patched_import(name, *args, **kwargs):
    # First import the module normally
    module = _original_import(name, *args, **kwargs)
    
    # If it's wx.core or wx._core, patch it
    if name in ('wx.core', 'wx._core', 'wx'):
        # Find _core module
        core_module = module
        if name == 'wx':
            if hasattr(module, '_core'):
                core_module = module._core
        
        # Patch the function directly
        if hasattr(core_module, '_macIsRunningOnMainDisplay'):
            print(f"[PATCH] Successfully patched {name}._macIsRunningOnMainDisplay")
            core_module._macIsRunningOnMainDisplay = _fake_mac_display_check
    
    return module

# Replace the built-in import function
__builtins__['__import__'] = _patched_import

# Ensure this patched module is only loaded once
sys.modules['wx_patch'] = sys.modules[__name__]
