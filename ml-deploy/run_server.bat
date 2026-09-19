@echo off
rem 一键本地起服务（用项目自带 venv，不碰 conda 全局环境）
setlocal
set VENV=D:\ML_Projects\ml-tutorial\.venv
if "%DEPLOY_MODEL_PATH%"=="" set DEPLOY_MODEL_PATH=D:\ML_Projects\ml-tutorial\ml-text\models\text_clf.joblib
"%VENV%\Scripts\python.exe" -m uvicorn deploy.service:app --app-dir src --host 127.0.0.1 --port 8000
