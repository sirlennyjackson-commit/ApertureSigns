@echo off
cd /d "%~dp0"
echo.
echo ==========================================
echo       APERTURE SIGNS UI v1
echo ==========================================
echo.
if not exist ".venv\Scripts\python.exe" (
    echo Project environment not found.
    echo Follow the setup instructions in README.md first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" app_ui_v1.py
if errorlevel 1 (
    echo.
    echo UI failed to start. Check the error above.
)
pause
