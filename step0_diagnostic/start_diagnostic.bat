@echo off
rem Double-click entry for the stage-0 diagnostic paper.
rem Keep this file ASCII + CRLF: cmd reads the file in the OEM codepage, so Chinese here
rem would print as mojibake. The real wording lives in run_diagnostic.py (UTF-8), which is
rem what actually prints to the screen.
chcp 65001 >nul
set PYTHONUTF8=1
set PY=D:\ML_Projects\ml-tutorial\.venv\Scripts\python.exe
if exist "%PY%" goto :run
where python >/dev/null 2>nul
if errorlevel 1 (set PY=py) else (set PY=python)
:run
"%PY%" -u "%~dp0run_diagnostic.py" %*
echo.
echo [end] This window waits for you - press any key to close it.
pause >nul
