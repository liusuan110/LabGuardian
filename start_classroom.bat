@echo off
REM ============================================================
REM  LabGuardian Classroom - One-Click Launcher (Windows)
REM  Starts the teacher server + student GUI in classroom mode
REM ============================================================

setlocal

set SCRIPT_DIR=%~dp0
set LABGUARDIAN_ROOT=%SCRIPT_DIR%

REM --- Default settings (override via .env or command line) ---
if "%LG_STATION_ID%"=="" set LG_STATION_ID=%COMPUTERNAME%
if "%LG_SERVER_PORT%"=="" set LG_SERVER_PORT=8080
if "%LG_SERVER_URL%"=="" set LG_SERVER_URL=http://localhost:%LG_SERVER_PORT%

echo.
echo  ========================================
echo   LabGuardian Classroom Mode
echo  ========================================
echo   Station ID : %LG_STATION_ID%
echo   Server URL : %LG_SERVER_URL%
echo   Server Port: %LG_SERVER_PORT%
echo  ========================================
echo.

REM --- Check Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+ and add to PATH.
    pause
    exit /b 1
)

REM --- Check if teacher frontend is built ---
if not exist "%LABGUARDIAN_ROOT%teacher\frontend\dist\index.html" (
    echo [INFO] Teacher frontend not built yet.
    echo [INFO] Checking for Node.js...
    node --version >nul 2>&1
    if errorlevel 1 (
        echo [WARN] Node.js not found. Teacher dashboard will show API-only mode.
        echo [WARN] Install Node.js and run: cd teacher\frontend ^&^& npm install ^&^& npm run build
    ) else (
        echo [INFO] Building teacher frontend...
        cd "%LABGUARDIAN_ROOT%teacher\frontend"
        if not exist "node_modules" (
            call npm install
        )
        call npm run build
        cd "%LABGUARDIAN_ROOT%"
        echo [INFO] Frontend built successfully.
    )
)

REM --- Launch ---
echo [INFO] Starting LabGuardian in classroom mode (hub)...
echo [INFO] Teacher dashboard: http://localhost:%LG_SERVER_PORT%
echo [INFO] Press Ctrl+C to stop.
echo.

python "%LABGUARDIAN_ROOT%src_v2\launcher.py" --classroom --hub --station-id %LG_STATION_ID%

pause
