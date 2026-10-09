@echo off
chcp 65001 >nul
cd /d "%~dp0"
title gemopt - check 3ds Max

echo.
echo Checks whether this program can launch 3ds Max.
echo.

python -m gemopt maxcheck

echo.
pause
