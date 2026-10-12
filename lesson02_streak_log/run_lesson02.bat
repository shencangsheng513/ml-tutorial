@echo off
setlocal EnableExtensions
title Lesson 02 - study log (saves to disk)
set "HERE=%~dp0"
set "FILE=%HERE%streak_log.py"
set "PY=%HERE%..\.venv\Scripts\python.exe"
if exist "%PY%" goto :pick
set "PY=python"
where python >nul 2>nul
if errorlevel 1 set "PY=py"
:pick
if exist "%FILE%" goto :run
echo [STOP] missing lesson file: %FILE%
echo [STOP] this .bat points at itself minus the drive letter. Do not move one without the other.
pause >nul
exit /b 1
:run
echo [py] running with: "%PY%"
echo.
"%PY%" -X utf8 -u "%FILE%"
set "RUN_RC=%ERRORLEVEL%"
echo.
echo [run] python exit code = %RUN_RC%
echo [note] exit code is NOT the verdict. The verdict is the [OK] / [FAIL] / [STOP] lines above.
echo [note] the log is one local file under LOCALAPPDATA, outside this repo. Nothing is uploaded.
pause >nul
exit /b 0
