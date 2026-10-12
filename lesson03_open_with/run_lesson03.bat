@echo off
setlocal EnableExtensions
title Lesson 03 - open/with buffering demo
set "HERE=%~dp0"
set "MAIN=%HERE%open_with.py"
set "CHILD=%HERE%write_demo_child.py"
set "PY=%HERE%..\.venv\Scripts\python.exe"
if exist "%PY%" goto :pick
set "PY=python"
where python >nul 2>nul
if errorlevel 1 set "PY=py"
:pick
if exist "%MAIN%" goto :check_child
echo [STOP] missing main file: %MAIN%
echo [STOP] this .bat points at itself minus the drive letter. Do not move one without the other.
pause >nul
exit /b 1
:check_child
if exist "%CHILD%" goto :run
echo [STOP] missing child file: %CHILD%
echo [STOP] this lesson needs both .py files. The .bat only launches the main one.
pause >nul
exit /b 1
:run
echo [py] running with: "%PY%"
echo.
"%PY%" -X utf8 -u "%MAIN%"
set "RUN_RC=%ERRORLEVEL%"
echo.
echo [run] python exit code = %RUN_RC%
echo [note] exit code is NOT the verdict. The verdict is the [OK] / [FAIL] / [STOP] lines above.
echo [note] everything written is inside your own temp folder and deleted at [7/7]. Nothing is uploaded.
pause >nul
exit /b 0
