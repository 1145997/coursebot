@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo 正在安装依赖，请稍候...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo 安装结束。按任意键退出。
pause >nul
