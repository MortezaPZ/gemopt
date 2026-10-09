@echo off
chcp 65001 >nul
cd /d "%~dp0"
title gemopt - build Max file

set "GEM=%~1"
if "%GEM%"=="" set "GEM=results\app\stone.gem"

if not exist "%GEM%" (
  echo.
  echo File "%GEM%" was not found.
  echo Finish an optimization first, then run this again.
  echo.
  pause
  exit /b 1
)

python -m gemopt tomax "%GEM%"

echo.
pause
