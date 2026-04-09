# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# 收集gradio的所有数据文件和子模块
gradio_datas = collect_data_files('gradio')
gradio_datas += collect_data_files('gradio_client')
gradio_datas += collect_data_files('safehttpx')
gradio_datas += collect_data_files('groovy')
# gradio需要运行时访问.py源文件（用于动态生成.pyi）
gradio_datas += collect_data_files('gradio', include_py_files=True)
gradio_datas += collect_data_files('gradio_client', include_py_files=True)
gradio_hidden = collect_submodules('gradio')
gradio_hidden += collect_submodules('gradio_client')
gradio_hidden += collect_submodules('safehttpx')
gradio_hidden += collect_submodules('groovy')

a = Analysis(
    ['ui_demo.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('models/nested_char_model.pth', 'models'),
        ('models/training_history.png', 'models'),
        ('models/test_results.txt', 'models'),
    ] + gradio_datas,
    hiddenimports=[
        'torch',
        'torchvision',
        'torchvision.transforms',
        'cv2',
        'PIL',
        'numpy',
        'nested_char_detector',
        'NestCharacter',
        'training_data_generator',
    ] + gradio_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tensorflow', 'tensorboard', 'keras'],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NestedCharDemo',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='NestedCharDemo',
)
