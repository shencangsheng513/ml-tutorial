@echo off
rem Double-click loop for the stage-0 paper, Q1 item 4 (append_record).
rem Keep this file ASCII + CRLF: cmd parses it in the OEM codepage, so Chinese here
rem prints as mojibake. All Chinese wording on screen comes from q1_basics.py itself.
rem
rem What it does: prints the line you are about to change, opens the editor parked on
rem that line, WAITS until you close the file, then prints the same readings again plus
rem the paper's own score. It never writes to the paper - the two lines are yours.
rem
rem Step [3/4] is the one added in r44: the editor command's own rc used to be thrown
rem away, so 'editor failed to launch' and 'you closed it without saving' printed the
rem same words. rc is still never treated as proof - it is reported next to the two
rem readings (mtime + size) that actually decide whether bytes landed.
chcp 65001 >nul
setlocal
set FILE=D:\ML_Projects\ml-tutorial\step0_diagnostic\q1_basics.py
set PY=D:\ML_Projects\ml-tutorial\.venv\Scripts\python.exe
set CODE=C:\Users\shenh\AppData\Local\Programs\Microsoft VS Code\bin\code.cmd
if not exist "%FILE%" (echo [STOP] paper file not found: "%FILE%" & goto :end)
if not exist "%PY%" (echo [STOP] venv python not found: "%PY%" & goto :end)

echo [1/4] BEFORE - the file as it stands right now
for %%F in ("%FILE%") do set "BEFORE_M=%%~tF"
for %%F in ("%FILE%") do set "BEFORE_S=%%~zF"
echo        mtime=%BEFORE_M%   size=%BEFORE_S% bytes
findstr /N /R "^" "%FILE%" | findstr /R "^43:"
echo.
echo [2/4] opening the editor, cursor on line 43.
echo        You need a SENTINEL default in the signature and your own check
echo        before the append line. Save with Ctrl+S.
echo.
echo        THIS WINDOW WILL WAIT until you close that file in the editor.
echo        Waiting is not a hang: Ctrl+S, then close the file tab or the window.
set EDITOR_RC=skipped
if exist "%CODE%" goto :use_code
echo        [WARN] VS Code launcher not at "%CODE%" - using notepad instead.
start "" /wait notepad "%FILE%"
goto :opened
:use_code
call "%CODE%" --wait --goto "%FILE%":43
set EDITOR_RC=zero
if errorlevel 1 set EDITOR_RC=nonzero
:opened

echo.
echo [3/4] WHAT THE EDITOR STEP ACTUALLY DID
if "%EDITOR_RC%"=="nonzero" echo        [WARN] the editor command came back with a nonzero rc, so the file
if "%EDITOR_RC%"=="nonzero" echo               may never have opened. That is a report, not a verdict: the
if "%EDITOR_RC%"=="nonzero" echo               readings in [4/4] are what decide.
if "%EDITOR_RC%"=="zero" echo        [ok] the editor command came back with rc=0. rc alone proves
if "%EDITOR_RC%"=="zero" echo        nothing (a broken .bat returns 0 too) - see [4/4].
if "%EDITOR_RC%"=="skipped" echo        [note] the editor command was skipped, notepad was used instead.

echo.
echo [4/4] AFTER - the same two readings, what they mean, then the score
for %%F in ("%FILE%") do set "AFTER_M=%%~tF"
for %%F in ("%FILE%") do set "AFTER_S=%%~zF"
echo        mtime=%AFTER_M%   size=%AFTER_S% bytes
findstr /N /R "^" "%FILE%" | findstr /R "^43:"
echo.
if "%AFTER_M%"=="%BEFORE_M%" if "%AFTER_S%"=="%BEFORE_S%" goto :unchanged
echo        [CHANGED] file was written. before: mtime=%BEFORE_M% size=%BEFORE_S%
echo                          after : mtime=%AFTER_M% size=%AFTER_S%
goto :score
:unchanged
echo        [UNCHANGED] mtime and size are identical to [1/4], so no bytes landed
echo        since the BEFORE reading. Resolution note: mtime here is minute-wide, so
echo        this line says 'nothing was written', it does not say why.
:score
echo.
echo        The paper's own score:
"%PY%" -X utf8 -u "%FILE%"
echo.
echo [note] Only the fourth item can move, and it moves like this:
echo        [FAIL] becomes [OK], and Q1: 0/4 becomes Q1: 1/4.
echo        The other three print [TODO] on purpose while they are empty.

:end
echo.
echo This window waits for you - press any key to close it.
pause >nul
endlocal
exit /b 0
