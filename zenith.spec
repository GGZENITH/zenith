# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:\\Users\\thoma_737ag4c\\Vast\\src\\zenith.py'],
    pathex=['C:\\Users\\thoma_737ag4c\\Vast'],
    binaries=[],
    datas=[('C:\\Users\\thoma_737ag4c\\Vast\\src\\models', 'models'), ('C:\\Users\\thoma_737ag4c\\Vast\\yolov8', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='zenith',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
