@echo off
chcp 65001 >nul
cd /d "%~dp0"
title gemopt - بررسی نصب مکس

echo.
echo این بررسی می‌کند که آیا برنامه می‌تواند 3ds Max را خودکار اجرا کند یا نه.
echo.

python -m gemopt maxcheck

echo.
pause
