@echo off
REM =====================================================================
REM Intelligent Attendance Monitoring System - Windows uninstaller
REM =====================================================================
REM Removes what setup.bat installed and what run.bat created, one part
REM at a time:
REM   - in the project folder: .venv with every Python library, only the
REM     optional dlib recogniser, the temporary files, your data, or the
REM     whole IAMS folder
REM   - shared with the rest of the PC: the Microsoft C++ build tools,
REM     CMake, Python 3.10 and the pip / Streamlit / Matplotlib caches
REM
REM Every part is asked separately, nothing is removed before the final
REM summary is confirmed, and your data is only deleted after DELETE is
REM typed. setup.bat records the prerequisites it installs in
REM .iams_installed, so shared software that was already on this PC is
REM marked as such.
REM
REM Exit codes:
REM   0 = finished, or nothing was selected
REM   1 = stopped, or a part could not be removed
REM   2 = finished - a restart completes removing the C++ build tools
REM
REM Design note: the same rules as setup.bat - goto/labels and
REM single-line IF statements only, no parentheses in echo text.
REM =====================================================================

setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%~dp0"
set "APPDIR=%ROOT:~0,-1%"
set "HELPER=%ROOT%scripts\setup_helper.py"
set "RECORD=%ROOT%.iams_installed"
set "VENV_PY=%ROOT%.venv\Scripts\python.exe"
set "ENVDIR=%TEMP%\iams_setup"
set "VS_ENV=%ENVDIR%\vs_list.bat"
set "PIP_CACHE=%LOCALAPPDATA%\pip\Cache"
set "ST_DIR=%USERPROFILE%\.streamlit"
set "MPL_DIR=%USERPROFILE%\.matplotlib"

set "DO_APP="
set "DO_VENV="
set "DO_DLIB="
set "DO_CACHE="
set "DO_DATA="
set "DO_CMAKE="
set "DO_PYTHON="
set "DO_SHARED_CACHE="
set "ANY_VS="
set "IAMS_VS_COUNT=0"
set "IAMS_VS_INSTALLER="
set "ANS="
set "FAILED="
set "REBOOT="

echo.
echo ============================================================
echo  Intelligent Attendance Monitoring System - Uninstall
echo ============================================================
echo.
echo This removes IAMS from this PC, part by part:
echo.
echo   1. The project folder: the Python libraries in .venv, the
echo      temporary files, your data - or the whole folder
echo   2. Software shared with the rest of the PC: the Microsoft C++
echo      build tools, CMake, Python 3.10 and shared caches
echo   3. A summary - nothing is removed before you confirm it
echo.

if not exist "%ROOT%app.py" goto :not_project
if not exist "%ROOT%config.py" goto :not_project
if not exist "%ENVDIR%" mkdir "%ENVDIR%" >nul 2>&1

REM Any Python can run the helper - it only uses the standard library.
set "PY_CMD="
py -3 -c "pass" >nul 2>&1
if not errorlevel 1 set "PY_CMD=py -3"
if defined PY_CMD goto :python_checked
python -c "pass" >nul 2>&1
if not errorlevel 1 set "PY_CMD=python"
:python_checked

set "HAVE_WINGET="
where winget >nul 2>&1
if not errorlevel 1 set "HAVE_WINGET=1"

set "REC_PYTHON="
set "REC_CMAKE="
set "REC_MSVC="
if not exist "%RECORD%" goto :records_read
findstr /x /c:"python" "%RECORD%" >nul 2>&1
if not errorlevel 1 set "REC_PYTHON=1"
findstr /x /c:"cmake" "%RECORD%" >nul 2>&1
if not errorlevel 1 set "REC_CMAKE=1"
findstr /x /c:"msvc-buildtools" /c:"msvc-workload" "%RECORD%" >nul 2>&1
if not errorlevel 1 set "REC_MSVC=1"
:records_read

REM =====================================================================
REM [1/3] The project folder
REM =====================================================================
echo ------------------------------------------------------------
echo [1/3] The IAMS project folder
echo ------------------------------------------------------------
echo.
echo Folder: %APPDIR%
set "HAVE_VENV="
if exist "%ROOT%.venv\" set "HAVE_VENV=1"
call :count_data
echo.
echo You can remove the WHOLE folder - the app itself, .venv with
echo every Python library, the temporary files and your data - or
echo answer N and choose the parts one by one.
echo.
set "ANS="
set /p ANS="Remove the whole IAMS folder? (Y/N) "
if /i not "%ANS%"=="Y" goto :ask_venv
echo.
echo WARNING - every file in this folder is deleted:
echo   %APPDIR%
if not "%DATA_FILES%"=="0" echo This includes YOUR DATA: %DATA_FILES% files - the attendance
if not "%DATA_FILES%"=="0" echo database, the face encodings and photos of every enrolled
if not "%DATA_FILES%"=="0" echo student, snapshots and logs. Export any reports you need first.
if exist "%ROOT%.env" echo Your settings and API key in .env are deleted too.
if exist "%ROOT%.git\" echo This is also a git repository: commits and branches that were
if exist "%ROOT%.git\" echo not pushed are lost.
echo This cannot be undone.
echo.
set "ANS="
set /p ANS="Type DELETE to confirm, or press Enter to choose the parts instead: "
if "%ANS%"=="DELETE" goto :app_confirmed
echo.
echo Not confirmed - the folder is kept. Choose the parts one by one.
goto :ask_venv

:app_confirmed
set "DO_APP=1"
goto :step_shared

:ask_venv
if not defined HAVE_VENV goto :ask_cache
call :folder_size "%ROOT%.venv"
echo.
echo Virtual environment .venv - %SZ%
echo   Every Python library setup.bat installed for IAMS: streamlit,
echo   opencv, mediapipe, numpy, pandas and the rest, plus the
echo   optional dlib recogniser if it was installed. run.bat needs it,
echo   and setup.bat can rebuild it at any time.
set "ANS="
set /p ANS="Remove .venv? (Y/N) "
if /i "%ANS%"=="Y" set "DO_VENV=1"
if defined DO_VENV goto :ask_cache

if not exist "%HELPER%" goto :ask_cache
"%VENV_PY%" "%HELPER%" check-dlib --quiet >nul 2>&1
if errorlevel 1 goto :ask_cache
echo.
echo Optional dlib recogniser inside .venv
echo   face_recognition, face_recognition_models and dlib. IAMS
echo   recognises faces with OpenCV SFace and keeps working without
echo   them.
set "ANS="
set /p ANS="Remove only the dlib recogniser from .venv? (Y/N) "
if /i "%ANS%"=="Y" set "DO_DLIB=1"

:ask_cache
echo.
echo Temporary files
echo   The Python caches __pycache__ and .pytest_cache in the project
echo   folder, and the working files setup.bat keeps in
echo   %ENVDIR%
echo   They are rebuilt automatically when needed.
set "ANS="
set /p ANS="Remove the temporary files? (Y/N) "
if /i "%ANS%"=="Y" set "DO_CACHE=1"

:ask_data
if "%DATA_FILES%"=="0" if not exist "%ROOT%.env" goto :step_shared
echo.
echo YOUR DATA - %DATA_FILES% files in the data folder
echo   attendance database        data\database
echo   face encodings             data\face_encodings
echo   enrollment photos          data\enrollment_images
echo   unknown-face captures      data\unknown_faces
echo   attendance snapshots       data\attendance_snapshots
echo   logs                       data\logs
if exist "%ROOT%.env" echo   settings and API key       .env
echo   The biometric data of every enrolled student is deleted for
echo   good and cannot be recovered. Export any reports you need first.
echo.
set "ANS="
set /p ANS="Type DELETE to delete your data, or press Enter to keep it: "
if "%ANS%"=="DELETE" set "DO_DATA=1"
if not defined DO_DATA echo Your data is kept.

REM =====================================================================
REM [2/3] Software shared with the rest of the PC
REM =====================================================================
:step_shared
echo.
echo ------------------------------------------------------------
echo [2/3] Software shared with the rest of this PC
echo ------------------------------------------------------------
echo.
echo These are installed for the whole PC or your Windows user, and
echo other programs may use them too. Each one says whether setup.bat
echo installed it - remove the others only if nothing else needs them.
if not exist "%RECORD%" echo setup.bat has no record of installing anything on this PC.

echo.
echo Microsoft C++ build tools - only the optional dlib recogniser
echo needed them.
if not defined PY_CMD goto :vs_unlisted
if not exist "%HELPER%" goto :vs_unlisted
del "%VS_ENV%" >nul 2>&1
%PY_CMD% "%HELPER%" list-vs --quiet --env-file "%VS_ENV%" >nul 2>&1
if exist "%VS_ENV%" call "%VS_ENV%"
if "%IAMS_VS_COUNT%"=="0" echo   Not installed - nothing to remove.
if "%IAMS_VS_COUNT%"=="0" goto :ask_cmake
if not defined IAMS_VS_INSTALLER goto :vs_unlisted
if defined REC_MSVC echo   setup.bat installed C++ build tools on this PC.
if not defined REC_MSVC echo   setup.bat did not install these - they were already on this
if not defined REC_MSVC echo   PC or were installed by hand.
set /a VS_I=0

:vs_ask_loop
set /a VS_I+=1
if %VS_I% GTR %IAMS_VS_COUNT% goto :ask_cmake
call set "VS_PATH=%%IAMS_VS_%VS_I%_PATH%%"
call set "VS_NAME=%%IAMS_VS_%VS_I%_NAME%%"
call set "VS_KIND=%%IAMS_VS_%VS_I%_KIND%%"
echo.
echo   %VS_NAME%
echo   %VS_PATH%
if "%VS_KIND%"=="product" goto :vs_ask_product
echo   Removes only the workload "Desktop development with C++" - the
echo   rest of Visual Studio stays, but it can no longer build C++.
set "ANS="
set /p ANS="  Remove the C++ workload from this Visual Studio? (Y/N) "
goto :vs_answer

:vs_ask_product
echo   Build Tools exist only to compile C++, so the whole product is
echo   uninstalled - this frees roughly 3 to 7 GB.
set "ANS="
set /p ANS="  Uninstall these Build Tools? (Y/N) "

:vs_answer
if /i not "%ANS%"=="Y" goto :vs_ask_loop
set "IAMS_VS_%VS_I%_DO=1"
set "ANY_VS=1"
goto :vs_ask_loop

:vs_unlisted
echo   They cannot be listed from here. If you want to remove them,
echo   open the Visual Studio Installer, choose Modify and untick
echo   "Desktop development with C++", or uninstall the Build Tools.

:ask_cmake
echo.
echo CMake - only the optional dlib recogniser needed it.
if not defined HAVE_WINGET goto :no_winget
set "FOUND="
winget list --id Kitware.CMake --exact --accept-source-agreements >nul 2>&1
if not errorlevel 1 set "FOUND=1"
if not defined FOUND echo   No separately installed CMake was found - nothing to remove.
if not defined FOUND goto :ask_python
if defined REC_CMAKE echo   setup.bat installed it.
if not defined REC_CMAKE echo   setup.bat did not install it - it was already on this PC or
if not defined REC_CMAKE echo   was installed by hand.
set "ANS="
set /p ANS="  Uninstall CMake? (Y/N) "
if /i "%ANS%"=="Y" set "DO_CMAKE=1"

:ask_python
echo.
echo Python 3.10 - the interpreter IAMS runs on.
set "FOUND="
winget list --id Python.Python.3.10 --exact --accept-source-agreements >nul 2>&1
if not errorlevel 1 set "FOUND=1"
if not defined FOUND echo   Not installed - nothing to remove.
if not defined FOUND goto :ask_shared_cache
if defined REC_PYTHON echo   setup.bat installed it.
if not defined REC_PYTHON echo   setup.bat did not install it - it was already on this PC or
if not defined REC_PYTHON echo   was installed by hand.
echo   Other Python programs and scripts on this PC may use it too.
if defined HAVE_VENV if not defined DO_VENV if not defined DO_APP echo   A kept .venv built from it stops working - setup.bat rebuilds it.
set "ANS="
set /p ANS="  Uninstall Python 3.10? (Y/N) "
if /i "%ANS%"=="Y" set "DO_PYTHON=1"
goto :ask_shared_cache

:no_winget
echo   winget is not available, so CMake and Python 3.10 cannot be
echo   checked or removed from here. Remove them in Windows Settings,
echo   Apps, Installed apps.

:ask_shared_cache
if not exist "%PIP_CACHE%\" if not exist "%ST_DIR%\" if not exist "%MPL_DIR%\" goto :summary
echo.
echo Shared caches in your user profile
if exist "%PIP_CACHE%\" call :folder_size "%PIP_CACHE%"
if exist "%PIP_CACHE%\" echo   pip download cache - %SZ%   %PIP_CACHE%
if exist "%ST_DIR%\" echo   Streamlit settings          %ST_DIR%
if exist "%MPL_DIR%\" echo   Matplotlib font cache       %MPL_DIR%
echo   Every Python project on this PC shares them. They are rebuilt
echo   when needed, but Streamlit settings you made yourself are lost.
set "ANS="
set /p ANS="Remove these caches? (Y/N) "
if /i "%ANS%"=="Y" set "DO_SHARED_CACHE=1"

REM =====================================================================
REM [3/3] Summary
REM =====================================================================
:summary
echo.
echo ------------------------------------------------------------
echo [3/3] Summary
echo ------------------------------------------------------------
echo.
set "ANY="
if defined DO_APP set "ANY=1"
if defined DO_VENV set "ANY=1"
if defined DO_DLIB set "ANY=1"
if defined DO_CACHE set "ANY=1"
if defined DO_DATA set "ANY=1"
if defined ANY_VS set "ANY=1"
if defined DO_CMAKE set "ANY=1"
if defined DO_PYTHON set "ANY=1"
if defined DO_SHARED_CACHE set "ANY=1"
if not defined ANY goto :nothing_selected

echo These will be removed:
if defined DO_APP echo   - the whole IAMS folder with everything in it
if defined DO_VENV echo   - the virtual environment .venv with every Python library
if defined DO_DLIB echo   - the dlib recogniser from .venv
if defined DO_CACHE echo   - the temporary files
if defined DO_DATA echo   - YOUR DATA in the data folder and .env
if defined ANY_VS call :summary_vs
if defined DO_CMAKE echo   - CMake
if defined DO_PYTHON echo   - Python 3.10
if defined DO_SHARED_CACHE echo   - the shared pip, Streamlit and Matplotlib caches
echo.
echo IAMS is closed first if it is running. Windows may ask for
echo administrator approval to remove the shared software.
echo.
set "ANS="
set /p ANS="Start removing now? (Y/N) "
if /i not "%ANS%"=="Y" goto :cancelled

REM =====================================================================
REM Removal
REM =====================================================================
echo.
echo ------------------------------------------------------------
echo Removing
echo ------------------------------------------------------------
if defined DO_APP goto :run_stop
if defined DO_VENV goto :run_stop
if defined DO_DLIB goto :run_stop
if defined DO_DATA goto :run_stop
goto :run_dlib

:run_stop
echo.
echo Closing IAMS if it is running ...
set "IAMS_VENV=%ROOT%.venv\"
powershell -NoProfile -Command "$p = @(Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($env:IAMS_VENV, [System.StringComparison]::OrdinalIgnoreCase) }); foreach ($x in $p) { Write-Host ('  Closing ' + $x.Name + ', process ' + $x.ProcessId); taskkill /F /T /PID $x.ProcessId 2>$null | Out-Null }; if ($p.Count -eq 0) { Write-Host '  IAMS is not running.' }"

:run_dlib
if defined DO_APP goto :run_shared_cache
if not defined DO_DLIB goto :run_venv
echo.
echo Removing the dlib recogniser from .venv ...
"%VENV_PY%" -m pip uninstall -y face-recognition face-recognition-models dlib
if errorlevel 1 call :failed "The dlib recogniser could not be fully removed - see above."

:run_venv
if not defined DO_VENV goto :run_cache
echo.
echo Removing .venv ...
rmdir /s /q "%ROOT%.venv"
if exist "%ROOT%.venv\" call :failed ".venv could not be fully removed - close every program that uses it and run uninstall.bat again."
if not exist "%ROOT%.venv\" echo   Done.

:run_cache
if not defined DO_CACHE goto :run_data
echo.
echo Removing the temporary files ...
if exist "%ROOT%.pytest_cache\" rmdir /s /q "%ROOT%.pytest_cache"
set "IAMS_ROOT=%ROOT%"
powershell -NoProfile -Command "$d = @(Get-ChildItem -LiteralPath $env:IAMS_ROOT -Directory -Force | Where-Object { $_.Name -notin @('.venv', '.git') }); $p = @($d | Where-Object { $_.Name -eq '__pycache__' }) + @($d | ForEach-Object { Get-ChildItem -LiteralPath $_.FullName -Directory -Recurse -Force -Filter __pycache__ -ErrorAction SilentlyContinue }); $p | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
echo   Done.

:run_data
if not defined DO_DATA goto :run_shared_cache
echo.
echo Deleting your data ...
set "IAMS_DATA=%ROOT%data"
if exist "%ROOT%data\" powershell -NoProfile -Command "Get-ChildItem -LiteralPath $env:IAMS_DATA -Recurse -Force -File | Where-Object { $_.Name -ne '.gitkeep' } | Remove-Item -Force -ErrorAction SilentlyContinue; Get-ChildItem -LiteralPath $env:IAMS_DATA -Recurse -Force -Directory | Sort-Object { $_.FullName.Length } -Descending | Where-Object { -not (Get-ChildItem -LiteralPath $_.FullName -Force) } | Remove-Item -Force -ErrorAction SilentlyContinue"
if exist "%ROOT%.env" del /f /q "%ROOT%.env"
call :count_data
if not "%DATA_FILES%"=="0" call :failed "%DATA_FILES% data files are in use and were kept - close every program showing them and run uninstall.bat again."
if exist "%ROOT%.env" call :failed ".env could not be deleted."
if "%DATA_FILES%"=="0" echo   Done.

:run_shared_cache
if not defined DO_SHARED_CACHE goto :run_vs
echo.
echo Removing the shared caches ...
if exist "%PIP_CACHE%\" rmdir /s /q "%PIP_CACHE%"
if exist "%ST_DIR%\" rmdir /s /q "%ST_DIR%"
if exist "%MPL_DIR%\" rmdir /s /q "%MPL_DIR%"
if exist "%PIP_CACHE%\" call :failed "The pip cache could not be fully removed."
echo   Done.

:run_vs
if not defined ANY_VS goto :run_cmake
set /a VS_I=0

:run_vs_loop
set /a VS_I+=1
if %VS_I% GTR %IAMS_VS_COUNT% goto :run_cmake
call set "VS_DO=%%IAMS_VS_%VS_I%_DO%%"
if not "%VS_DO%"=="1" goto :run_vs_loop
call set "VS_PATH=%%IAMS_VS_%VS_I%_PATH%%"
call set "VS_NAME=%%IAMS_VS_%VS_I%_NAME%%"
call set "VS_KIND=%%IAMS_VS_%VS_I%_KIND%%"
echo.
echo Removing the C++ tools from %VS_NAME%.
echo The Visual Studio Installer does this - it can take 10 to 30
echo minutes, and Windows asks for administrator approval ...
if "%VS_KIND%"=="product" "%IAMS_VS_INSTALLER%" uninstall --installPath "%VS_PATH%" --passive --norestart
if not "%VS_KIND%"=="product" "%IAMS_VS_INSTALLER%" modify --installPath "%VS_PATH%" --remove Microsoft.VisualStudio.Workload.NativeDesktop --passive --norestart
set "IAMS_RC=%ERRORLEVEL%"
if "%IAMS_RC%"=="3010" set "REBOOT=1"
if "%IAMS_RC%"=="3010" set "IAMS_RC=0"
if "%IAMS_RC%"=="1641" set "REBOOT=1"
if "%IAMS_RC%"=="1641" set "IAMS_RC=0"
if not "%IAMS_RC%"=="0" call :failed "The Visual Studio Installer returned exit code %IAMS_RC%."
if "%IAMS_RC%"=="0" echo   Done.
goto :run_vs_loop

:run_cmake
if not defined DO_CMAKE goto :run_python
echo.
echo Uninstalling CMake ...
winget uninstall --id Kitware.CMake --exact --accept-source-agreements
set "IAMS_RC=%ERRORLEVEL%"
if not "%IAMS_RC%"=="0" call :failed "winget could not uninstall CMake - remove it in Windows Settings, Apps."
if "%IAMS_RC%"=="0" call :unrecord cmake

:run_python
if not defined DO_PYTHON goto :run_finish
echo.
echo Uninstalling Python 3.10 ...
winget uninstall --id Python.Python.3.10 --exact --accept-source-agreements
set "IAMS_RC=%ERRORLEVEL%"
if not "%IAMS_RC%"=="0" call :failed "winget could not uninstall Python 3.10 - remove it in Windows Settings, Apps."
if "%IAMS_RC%"=="0" call :unrecord python

:run_finish
if defined DO_APP goto :run_app
if defined DO_CACHE if exist "%ENVDIR%\" rmdir /s /q "%ENVDIR%"
echo.
echo ------------------------------------------------------------
if defined FAILED echo Finished with problems - see the messages above.
if not defined FAILED echo Uninstall finished.
if defined REBOOT echo Restart Windows to complete removing the C++ build tools.
echo ------------------------------------------------------------
echo.
pause
if defined FAILED exit /b 1
if defined REBOOT exit /b 2
exit /b 0

:run_app
if exist "%ENVDIR%\" rmdir /s /q "%ENVDIR%"
echo.
echo ------------------------------------------------------------
echo Last step: deleting the IAMS folder
echo   %APPDIR%
echo ------------------------------------------------------------
if defined FAILED echo Some earlier parts had problems - see the messages above.
if defined REBOOT echo Restart Windows afterwards to complete removing the C++ build tools.
echo This window closes when the folder is gone.
echo.
pause
REM "(goto)" ends this batch file before the folder holding it is
REM deleted; the rest of the line still runs.
(goto) 2>nul & cd /d "%TEMP%" & rmdir /s /q "%APPDIR%" & if exist "%APPDIR%\" echo Some files were in use and are left in %APPDIR% - close the programs using them and delete the folder by hand. & pause
exit /b 0

REM =====================================================================
REM Subroutines and exits
REM =====================================================================
:summary_vs
set /a VS_I=0
:summary_vs_loop
set /a VS_I+=1
if %VS_I% GTR %IAMS_VS_COUNT% exit /b 0
call set "VS_DO=%%IAMS_VS_%VS_I%_DO%%"
call set "VS_NAME=%%IAMS_VS_%VS_I%_NAME%%"
call set "VS_KIND=%%IAMS_VS_%VS_I%_KIND%%"
if not "%VS_DO%"=="1" goto :summary_vs_loop
if "%VS_KIND%"=="product" echo   - %VS_NAME%
if not "%VS_KIND%"=="product" echo   - the C++ workload of %VS_NAME%
goto :summary_vs_loop

:folder_size
REM %1 = folder. Sets SZ to its size, e.g. "1.4 GB".
set "SZ=size unknown"
set "IAMS_SIZE_DIR=%~1"
if not exist "%ENVDIR%" mkdir "%ENVDIR%" >nul 2>&1
powershell -NoProfile -Command "$s = (Get-ChildItem -LiteralPath $env:IAMS_SIZE_DIR -Recurse -Force -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum; if ($s -ge 1GB) { '{0:N1} GB' -f ($s / 1GB) } else { '{0:N0} MB' -f ($s / 1MB) }" > "%ENVDIR%\size.txt" 2>nul
if exist "%ENVDIR%\size.txt" set /p "SZ=" < "%ENVDIR%\size.txt"
exit /b 0

:count_data
REM Sets DATA_FILES to the number of user data files under data\.
set "DATA_FILES=0"
if not exist "%ROOT%data\" exit /b 0
if not exist "%ENVDIR%" mkdir "%ENVDIR%" >nul 2>&1
set "IAMS_DATA=%ROOT%data"
powershell -NoProfile -Command "@(Get-ChildItem -LiteralPath $env:IAMS_DATA -Recurse -Force -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne '.gitkeep' }).Count" > "%ENVDIR%\count.txt" 2>nul
if exist "%ENVDIR%\count.txt" set /p "DATA_FILES=" < "%ENVDIR%\count.txt"
exit /b 0

:unrecord
REM %1 = a line of .iams_installed to drop after its software is gone.
if not exist "%RECORD%" exit /b 0
findstr /v /x /c:"%~1" "%RECORD%" > "%RECORD%.tmp"
move /y "%RECORD%.tmp" "%RECORD%" >nul
exit /b 0

:failed
REM %1 = what went wrong.
set "FAILED=1"
echo   PROBLEM: %~1
exit /b 0

:not_project
echo UNINSTALL STOPPED - uninstall.bat must stay in the IAMS project
echo folder, next to app.py and config.py. Run it from there.
goto :fail

:nothing_selected
echo Nothing was selected - nothing was changed.
echo.
pause
exit /b 0

:cancelled
echo.
echo Cancelled - nothing was changed.
echo.
pause
exit /b 0

:fail
echo.
pause
exit /b 1
