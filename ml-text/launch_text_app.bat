@echo off
rem 帖子归版助手启动脚本：8503 端口已活着就直接开浏览器，否则先拉起服务
powershell -NoProfile -Command ^
  "$alive=$false; try { Invoke-WebRequest -Uri http://localhost:8503 -UseBasicParsing -TimeoutSec 2 | Out-Null; $alive=$true } catch { if ($_.Exception.Response) { $alive=$true } };" ^
  "if (-not $alive) { Start-Process -FilePath 'C:\Users\shenh\miniconda3\python.exe' -ArgumentList '-m','streamlit','run','app.py','--server.port','8503' -WorkingDirectory 'D:\ML_Projects\ml-tutorial\ml-text' -WindowStyle Minimized; Start-Sleep -Seconds 8 };" ^
  "Start-Process 'http://localhost:8503'"
