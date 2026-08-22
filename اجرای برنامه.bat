@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Gem Cut Optimiser
python -m gemopt.app
if errorlevel 1 (
  echo.
  echo ---------------------------------------------------
  echo Something went wrong. Common causes:
  echo   * Python is not installed, or not on PATH
  echo   * dependencies missing: run  pip install -r requirements.txt
  echo ---------------------------------------------------
  pause
)
