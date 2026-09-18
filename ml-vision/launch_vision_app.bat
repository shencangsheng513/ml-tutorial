@echo off
rem 手绘数字识别 App 启动器：已在跑就直接开浏览器，没跑就后台拉起
powershell -NoProfile -Command ^
 "$p=8504; $up=$false; try{ Invoke-WebRequest -Uri ('http://localhost:'+$p) -TimeoutSec 2 -UseBasicParsing | Out-Null; $up=$true }catch{};" ^
 "if(-not $up){ Start-Process -FilePath 'C:\Users\shenh\miniconda3\python.exe' -ArgumentList '-m','streamlit','run','app.py','--server.port','8504','--server.headless','true' -WorkingDirectory 'D:\ML_Projects\ml-tutorial\ml-vision' -WindowStyle Minimized; Start-Sleep -Seconds 10 };" ^
 "Start-Process ('http://localhost:'+$p)"
