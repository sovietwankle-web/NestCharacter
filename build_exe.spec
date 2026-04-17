# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec - 嵌套字符工坊打包
仅打包核心功能：生成、检测、加密对抗
"""
import os
import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# 需要打包的数据文件
datas = [
    ('models/nested_char_model.pth', 'models'),
    ('models/nested_char_model_dual.pth', 'models'),
    ('training_data/ocr_data/char_vocab.json', 'training_data/ocr_data'),
]

# torch / torchvision 的数据文件（模型权重加载需要）
datas += collect_data_files('torch')
datas += collect_data_files('torchvision')

# 隐式导入（torch动态加载的子模块）
hiddenimports = [
    'torch',
    'torchvision',
    'torchvision.transforms',
    'torchvision.models',
    'cv2',
    'PIL',
    'numpy',
    'nested_char_detector',
    'NestCharacter',
]

# torch 子模块收集
hiddenimports += collect_submodules('torch')
hiddenimports += collect_submodules('torchvision')

# 排除不需要的大包（大幅缩小体积）
excludes = [
    'tensorflow', 'tensorboard', 'keras',
    'gradio', 'gradio_client',
    'fastapi', 'uvicorn', 'starlette',
    'requests', 'urllib3', 'httpx',
    'scipy', 'sklearn', 'scikit-learn', 'skimage', 'scikit-image',
    'seaborn', 'matplotlib', 'tqdm',
    'IPython', 'notebook', 'jupyter',
    'pyarrow', 'pandas',
    'tkinter.test',
    'api_server', 'test_api', 'demo_system', 'test_system',
    'training_data_generator', 'train_model',
]

a = Analysis(
    ['ui_demo.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NestedCharDemo',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name='NestedCharDemo',
)
