@echo off
title CardioGuard AI - Heart Disease Deep Learning Predictor
echo =====================================================================
echo  CardioGuard AI - Heart Disease Deep Learning Prediction System
echo  Based on neural-network-tensorflow-auc-0-9529.ipynb (AUC: 0.9529)
echo =====================================================================
echo.
echo Launching Streamlit web application on http://localhost:8501 ...
echo.
"%~dp0.venv\Scripts\streamlit.exe" run "%~dp0app.py"
pause
