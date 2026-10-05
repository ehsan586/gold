@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title ASTRA TRADER - SET PERSONAL API TOKEN

echo.
echo ASTRA personal API token setup
echo.
echo 1 = Generate a strong random token (recommended)
echo 2 = Enter my own token
set /p CHOICE=Choose 1 or 2: 

if "%CHOICE%"=="2" goto CUSTOM

for /f "delims=" %%A in ('py -3 -c "import secrets; print(secrets.token_urlsafe(32))" 2^>nul') do set "TOKEN=%%A"
if not defined TOKEN for /f "delims=" %%A in ('python -c "import secrets; print(secrets.token_urlsafe(32))" 2^>nul') do set "TOKEN=%%A"
if not defined TOKEN (
  echo Python was not found. Run run.bat once first, or enter a token manually.
  goto CUSTOM
)
goto SAVE

:CUSTOM
set /p TOKEN=Enter your token (16+ characters): 

:SAVE
if not defined TOKEN (
  echo No token entered.
  pause
  exit /b 1
)
if "%TOKEN%"=="change-me-to-a-long-random-string" (
  echo That placeholder is not allowed.
  pause
  exit /b 1
)
>"ASTRA_API_TOKEN.txt" echo %TOKEN%

if exist ".env" (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$p='.env'; $t=Get-Content $p -Raw; if($t -match '(?m)^GOLDBOT_API_TOKEN=.*$'){ $t=[regex]::Replace($t,'(?m)^GOLDBOT_API_TOKEN=.*$','GOLDBOT_API_TOKEN=%TOKEN%') } else { $t += [Environment]::NewLine+'GOLDBOT_API_TOKEN=%TOKEN%'+[Environment]::NewLine }; Set-Content -Path $p -Value $t -Encoding utf8"
) else (
  copy /y ".env.example" ".env" >nul
  >>".env" echo GOLDBOT_API_TOKEN=%TOKEN%
)

echo.
echo Token saved in ASTRA_API_TOKEN.txt and .env
 echo Use this SAME token in the dashboard. You will only need it once per browser/device.
echo.
pause
