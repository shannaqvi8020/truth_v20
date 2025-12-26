# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for TruthScript - Windows Only
# Bundles ffmpeg for standalone operation

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_all

block_cipher = None

# Collect Whisper and Tiktoken data files
whisper_datas = collect_data_files('whisper')
tiktoken_datas = collect_data_files('tiktoken')

hidden_imports = [
    'tiktoken',
    'tiktoken_ext',
    'tiktoken_ext.openai_public',
    'numba',
    'llvmlite',
    'torch',
    'torchaudio',
    'ffmpeg',
    'numpy',
    'regex',
    'tqdm',
    'huggingface_hub',
    'PIL',
    'PIL._tkinter_finder',
    'whisper',
    'whisper.normalizers',
    'whisper.tokenizer',
    'tkinter',
    'tkinter.ttk',
    'tkinter.filedialog',
    'tkinter.messagebox',
    '_tkinter',
    'more_itertools',
]

# Data files to include
all_datas = [
    ('transcribe.py', '.'),
    ('utils.py', '.'),
]

# Add ffmpeg binaries if they exist (created by GitHub Actions)
if os.path.isdir('ffmpeg_bin'):
    all_datas.append(('ffmpeg_bin', 'ffmpeg_bin'))

all_datas += whisper_datas + tiktoken_datas

a = Analysis(
    ['desktop_app.py'],
    pathex=[],
    binaries=[],
    datas=all_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hook.py'],
    excludes=[
        'matplotlib',
        'scipy',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'sphinx',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TruthScript',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='TruthScript',
)
