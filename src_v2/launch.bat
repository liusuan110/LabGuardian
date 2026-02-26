@echo off
chcp 65001 >nul 2>&1
title LabGuardian

echo ╔══════════════════════════════════════╗
echo ║      LabGuardian — 启动中...         ║
echo ╚══════════════════════════════════════╝
echo.

:: --- 定位脚本所在目录（launch.bat 放在 src_v2/ 内） ---
cd /d "%~dp0"

:: --- 寻找虚拟环境 ---
set PYTHON=
if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "PYTHON=%~dp0..\.venv\Scripts\python.exe"
) else if exist "%~dp0..\venv\Scripts\python.exe" (
    set "PYTHON=%~dp0..\venv\Scripts\python.exe"
) else (
    where python >nul 2>&1 && set "PYTHON=python"
)

if "%PYTHON%"=="" (
    echo [错误] 未找到 Python，请检查虚拟环境
    pause
    exit /b 1
)

echo Python: %PYTHON%
echo.

:: --- 启动（默认看门狗模式，评审推荐） ---
"%PYTHON%" launcher.py %*

echo.
echo LabGuardian 已退出。
pause
