@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_release.ps1"
if errorlevel 1 pause
