@echo off
rem Double-click loop for the stage-0 paper, Q1 item 4 (append_record).
rem Keep this file ASCII + CRLF: cmd parses it in the OEM codepage, so Chinese here
rem prints as mojibake. All Chinese wording on screen comes from q1_basics.py itself.
rem
rem What it does: prints the line you are about to change, opens the editor parked on
rem that line, WAITS until you close the file, then prints the same reading again plus
rem the paper's own score. It never writes to the paper - the two lines are yours.
chcp 65001 >nul
setlocal
set FILE=D:\ML_Projects\ml-tutorial\step0_diagnostic\q1_basics.py
set PY=D:\ML_Projects\ml-tutorial\.venv\Scripts\python.exe
set CODE=C:\Users\shenh\AppData\Local\Programs\Microsoft VS Code\bin\code.cmd
if not exist "%FILE%" (echo [STOP] paper file not found: "%FILE%" & goto :end)
if not exist "%PY%" (echo [STOP] venv python not found: "%PY%" & goto :end)

echo [1/3] BEFORE - the file as it stands right now
for %%F in ("%FILE%") do echo        mtime=%%~tF   size=%%~zF bytes
findstr /N /R "^" "%FILE%" | findstr /R "^43:"
echo.
echo [2/3] opening the editor, cursor on line 43.
echo        You need a SENTINEL default in the signature and your own check
echo        before the append line. Save with Ctrl+S.
echo.
echo        THIS WINDOW WILL WAIT until you close that file in the editor.
echo        Waiting is not a hang: Ctrl+S, then close the file tab or the window.
if exist "%CODE%" (
    call "%CODE%" --wait --goto "%FILE%":43
) else (
    echo        [WARN] VS Code launcher not at "%CODE%" - using notepad instead.
    start "" /wait notepad "%FILE%"
)

echo.
echo [3/3] AFTER - same two readings, then the paper's own score
for %%F in ("%FILE%") do echo        mtime=%%~tF   size=%%~zF bytes
findstr /N /R "^" "%FILE%" | findstr /R "^43:"
echo.
"%PY%" -X utf8 -u "%FILE%"
echo.
echo [note] If mtime and size above are unchanged, nothing was saved, so the score
echo        has to stay Q1: 0/4 - and that is still an honest reading.
echo        The other three items print [TODO] on purpose; only the fourth line
echo        should move from [FAIL] to [OK].

:end
echo.
echo This window waits for you - press any key to close it.
pause >nul
endlocal
exit /b 0
