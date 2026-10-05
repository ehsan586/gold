@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title ASTRA TRADER - STARTUP

where py.exe >nul 2>&1
if not errorlevel 1 goto USE_PY
where python.exe >nul 2>&1
if not errorlevel 1 goto USE_PYTHON

echo [ERROR] Python 3.10+ was not found in PATH.
goto FAILED

:USE_PY
py.exe -3 "%~dp0bootstrap.py"
set "EC=%ERRORLEVEL%"
goto DONE

:USE_PYTHON
python.exe "%~dp0bootstrap.py"
set "EC=%ERRORLEVEL%"
goto DONE

:FAILED
set "EC=1"

:DONE
echo.
if not "%EC%"=="0" (
    echo ============================================================
    echo ASTRA STARTUP FAILED
    echo ============================================================
    echo Exit code: %EC%
) else (
    echo ============================================================
    echo ASTRA STARTUP FINISHED
    echo Execution mode is SIMULATION only.
    echo ============================================================
)
echo.
pause
exit /b %EC%
