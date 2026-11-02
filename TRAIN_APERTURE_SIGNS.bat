@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Project environment not found.
    echo Follow the setup instructions in README.md first.
    pause
    exit /b 1
)
set /p "ZIP_PATH=Enter the full path to the landmark dataset ZIP: "
if "%ZIP_PATH%"=="" goto :fail
if not exist "%ZIP_PATH%" (
    echo Dataset ZIP not found: "%ZIP_PATH%"
    goto :fail
)
".venv\Scripts\python.exe" backend\training\prepare_dataset.py --zip "%ZIP_PATH%" --min-examples 5
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" backend\training\train_v2.py
if errorlevel 1 goto :fail
echo.
echo TRAINING COMPLETE.
pause
exit /b 0
:fail
echo.
echo TRAINING FAILED. Read the error above.
pause
exit /b 1
