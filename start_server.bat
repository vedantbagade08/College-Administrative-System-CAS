@echo off
echo ========================================
echo CAS Admin Dashboard - Flask Server
echo ========================================
echo.
echo Make sure you have installed required packages:
echo   pip install flask opencv-python numpy pymysql flask-cors
echo.
echo Starting server on http://localhost:5000
echo Access admin dashboard at: http://localhost:5000/admin
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.
cd /d %~dp0
python app.py
pause

