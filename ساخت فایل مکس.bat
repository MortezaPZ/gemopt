@echo off
chcp 65001 >nul
cd /d "%~dp0"
title gemopt - ساخت فایل مکس

set "GEM=%~1"
if "%GEM%"=="" set "GEM=results\app\stone.gem"

if not exist "%GEM%" (
  echo.
  echo فایل "%GEM%" پیدا نشد.
  echo اول یک بهینه‌سازی را کامل اجرا کنید، بعد این را بزنید.
  echo.
  pause
  exit /b 1
)

python -m gemopt tomax "%GEM%"

echo.
pause
