@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo                ASTRA TRADER - STOP
echo ============================================================

if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -m goldbot shutdown
) else (
    where py.exe >nul 2>&1
    if not errorlevel 1 (
        py.exe -3 -m goldbot shutdown
    ) else (
        python.exe -m goldbot shutdown
    )
)
set "EC=%ERRORLEVEL%"
if "%EC%"=="0" echo ASTRA shutdown request sent.
if not "%EC%"=="0" echo No running ASTRA instance was stopped (or Python environment is unavailable).

echo.
pause
exit /b %EC%
