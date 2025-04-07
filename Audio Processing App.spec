# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('wx_path.py', '.')],
    hiddenimports=['openai', 'wx', 'wx.adv'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['wx_path.py'],
    excludes=['torch', 'torchvision', 'torchaudio'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Audio Processing App',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Audio Processing App',
)
app = BUNDLE(
    coll,
    name='Audio Processing App.app',
    icon=None,
    bundle_identifier=None,
)
