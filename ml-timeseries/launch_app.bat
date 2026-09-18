@echo off
rem 备货助手启动器：若 8502 端口无服务则后台拉起 Streamlit，然后打开浏览器
powershell -NoProfile -Command "$up=$false; try { Invoke-WebRequest -Uri http://localhost:8502 -TimeoutSec 2 -UseBasicParsing | Out-Null; $up=$true } catch {}; if (-not $up) { Start-Process -FilePath 'C:\Users\shenh\miniconda3\python.exe' -ArgumentList '-m','streamlit','run','app.py','--server.headless','true','--server.port','8502' -WorkingDirectory 'D:\ML_Projects\ml-tutorial\ml-timeseries' -WindowStyle Minimized; Start-Sleep -Seconds 8 }; Start-Process 'http://localhost:8502'"
