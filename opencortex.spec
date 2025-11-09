from PyInstaller.utils.hooks import collect_all, collect_data_files
import mne
import os

# Collect ALL MNE files
mne_datas, mne_binaries, mne_hiddenimports = collect_all('mne')
brainflow_datas, brainflow_binaries, brainflow_hiddenimports = collect_all('brainflow')

# Collect imblearn data files
imblearn_datas = collect_data_files('imblearn')

block_cipher = None

a = Analysis(
    ['opencortex/__main__.py'],
    pathex=[],
    binaries=mne_binaries + brainflow_binaries,  # Add brainflow binaries
    datas=[
        ('opencortex/configs/*.yaml', 'opencortex/configs'),
        ('opencortex/model.onnx', 'models'),
        ('opencortex/pacnet_8e_stew.onnx', 'models')
    ] + mne_datas + brainflow_datas + imblearn_datas,  # Add imblearn data files
    hiddenimports=[
        'lazy_loader',
        'mne',
        'imblearn',  # Add imblearn
        'sklearn',   # imblearn depends on sklearn
    ] + mne_hiddenimports + brainflow_hiddenimports,  # Add brainflow hidden imports
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='OpenCortex',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='OpenCortex',
)