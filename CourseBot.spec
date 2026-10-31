# -*- mode: python ; coding: utf-8 -*-

import os

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

DEBUG_CONSOLE = os.environ.get("CB_CONSOLE") == "1"
APP_NAME = "CourseBotDebug" if DEBUG_CONSOLE else "CourseBot"

# RapidOCR 的 ONNX 模型、config.yaml 以及动态导入的识别器模块
datas = collect_data_files("rapidocr_onnxruntime")
hiddenimports = collect_submodules("rapidocr_onnxruntime")
datas += collect_data_files("uiautomation")
hiddenimports += collect_submodules("uiautomation")

# onnxruntime 的 DLL / provider 插件
binaries = collect_dynamic_libs("onnxruntime")
hiddenimports += ["onnxruntime.capi._pybind_state"]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "pandas",
        "scipy",
        "IPython",
        "notebook",
        "pytest",
        "torch",
        "tensorflow",
        "onnx",
        "numba",
        "llvmlite",
        "pygments",
        "lxml",
        "onnxruntime.tools",
        "onnxruntime.transformers",
        "onnxruntime.quantization",
        "onnxruntime.training",
    ],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=DEBUG_CONSOLE,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=["packaging/coursebot.ico"],
    version="packaging/version_info.txt",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)

# 某些依赖会把很旧的 VC++ 运行库（14.16）带进来，onnxruntime 1.31 在加载时会
# 直接 0xC0000005 崩溃。构建完成后用 Windows 自带的新版运行库覆盖 app-local DLL。
import shutil
from pathlib import Path

_internal_dir = Path(DISTPATH) / APP_NAME / "_internal"
_runtime_dlls = [
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll",
    "msvcp140_codecvt_ids.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "concrt140.dll",
    "ucrtbase.dll",
]
_system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
if _internal_dir.exists():
    for _name in _runtime_dlls:
        _src = _system32 / _name
        if _src.exists():
            shutil.copy2(_src, _internal_dir / _name)
            print(f"[CourseBot] refreshed runtime DLL: {_name}")
