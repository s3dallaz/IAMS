@echo off
REM =====================================================================
REM Intelligent Attendance Monitoring System - Windows launcher
REM =====================================================================
REM Starts the Streamlit application using the virtual environment that
REM setup.bat created. It does NOT install or modify anything, so it is
REM fast to start and safe to run repeatedly.
REM
REM Design note: single-line IF statements only, no parentheses in echo
REM text - see the note at the top of setup.bat.
REM =====================================================================

setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo  Intelligent Attendance Monitoring System
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" goto :no_venv
if not exist "app.py" goto :no_app

echo Starting the application - the browser opens automatically.
echo Address: http://localhost:8501
echo Stop:    press Ctrl+C in this window.
echo.

set "APP_RC=0"
REM run_app.py starts the session scheduler, then Streamlit - so scheduled
REM sessions start and close on time even when no page is open
".venv\Scripts\python.exe" run_app.py
set "APP_RC=%ERRORLEVEL%"

echo.
if not "%APP_RC%"=="0" echo The application exited with code %APP_RC% - see the output above.
echo Application stopped.
pause
exit /b %APP_RC%

:no_venv
echo ERROR: No virtual environment found at .venv
echo Run setup.bat first, then run this script again.
goto :fail

:no_app
echo ERROR: app.py was not found.
echo Run run.bat from inside the project folder - it switches to its
echo own directory automatically, so a missing app.py means the
echo download is incomplete. Re-extract the ZIP archive.
goto :fail

:fail
echo.
pause
exit /b 1
