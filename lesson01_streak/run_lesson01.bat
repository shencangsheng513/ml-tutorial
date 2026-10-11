@echo off
setlocal EnableExtensions
title Lesson 01 - change one number
set "HERE=%~dp0"
set "FILE=%HERE%lesson01_streak.py"
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
echo [note] the lesson text is the Chinese block above. Next run: type 4 at its first prompt.
pause >nul
exit /b 0
