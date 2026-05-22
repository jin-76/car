@echo off
setlocal
cd /d "%~dp0"

set "PROJECT_ROOT=%~dp0.."
set "PYTHONW=%PROJECT_ROOT%\.venv-gpu\pythonw.exe"
set "PYTHON=%PROJECT_ROOT%\.venv-gpu\python.exe"

if exist "%PYTHONW%" (
  start "" "%PYTHONW%" "%~dp0SII_Traffic_App.py"
) else if exist "%PYTHON%" (
  start "" "%PYTHON%" "%~dp0SII_Traffic_App.py"
) else (
  start "" py -3 "%~dp0SII_Traffic_App.py"
)

endlocal
