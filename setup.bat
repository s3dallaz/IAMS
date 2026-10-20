@echo off
REM =====================================================================
REM Intelligent Attendance Monitoring System - Windows setup
REM =====================================================================
REM Consent based installer:
REM   - nothing is installed without an explicit Y answer
REM   - every check is skipped when the tool is already present
REM   - external installers still ask Windows for administrator approval
REM
REM Exit codes:
REM   0 = setup finished successfully
REM   1 = stopped or failed - read the message and run setup.bat again
REM   2 = a restart or a new terminal is needed - then run setup.bat again
REM
REM Safe to run repeatedly. It only touches .venv and installed packages.
REM Databases, face encodings, configuration and student data are never
REM modified.
REM
REM Design note: goto/labels and single-line IF statements only - no
REM multi-line parenthesized IF blocks, and no parentheses in echo text.
REM A ')' inside echo text within an IF (...) block terminates the block
REM early and makes cmd.exe abort the whole batch file with
REM "X was unexpected at this time".
REM =====================================================================

setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%~dp0"
set "HELPER=%ROOT%scripts\setup_helper.py"
set "ENVDIR=%TEMP%\iams_setup"
set "CMAKE_ENV=%ENVDIR%\cmake_env.bat"
set "MSVC_ENV=%ENVDIR%\msvc_env.bat"

REM An RC environment variable (set by this script in older versions, or
REM inherited from Windows) makes CMake use it as the resource compiler and
REM the dlib build fails with: Could not find the compiler specified in the
REM environment variable RC. Clear it for this setup session only.
set "RC="

set "PY_CMD="
set "BUILD_NEEDED=1"
set "ANS="
set "IAMS_RC=0"
set "HAVE_WINGET="
set "IAMS_CMAKE_OK="
set "IAMS_CMAKE_BIN="
set "IAMS_MSVC_OK="
set "IAMS_VS_HAS_CPP="
set "IAMS_VS_INSTANCE="
set "IAMS_VS_INSTALLER="

if not exist "%ENVDIR%" mkdir "%ENVDIR%" >nul 2>&1

echo.
echo ============================================================
echo  Intelligent Attendance Monitoring System - Setup
echo ============================================================
echo.
echo Welcome. This setup checks this PC step by step:
echo.
echo   1. Windows 64-bit and free disk space
echo   2. Python - version 3.10 is recommended
echo   3. CMake - only needed when dlib has to be compiled
echo   4. Microsoft C++ build tools - only needed when dlib has
echo      to be compiled
echo   5. The project virtual environment .venv
echo   6. The Python packages from requirements.txt
echo   7. A final import check of every critical package
echo.
echo Nothing is installed without asking you first. External
echo installers may still ask for Windows administrator approval.
echo Your database, face encodings, configuration and student
echo data are never modified.
echo.

if not exist "%HELPER%" goto :helper_missing

REM =====================================================================
REM [1/7] Windows and Python
REM =====================================================================
echo ------------------------------------------------------------
echo [1/7] Checking Windows and looking for Python
echo ------------------------------------------------------------
echo.
if /i "%PROCESSOR_ARCHITECTURE%"=="x86" if not defined PROCESSOR_ARCHITEW6432 goto :unsupported_windows

set "PY_CMD="
py -3.10 "%HELPER%" check-python --quiet >nul 2>&1
if not errorlevel 1 set "PY_CMD=py -3.10"
if defined PY_CMD goto :python_found

py -3.11 "%HELPER%" check-python --quiet >nul 2>&1
if not errorlevel 1 set "PY_CMD=py -3.11"
if defined PY_CMD goto :python_found

py -3.12 "%HELPER%" check-python --quiet >nul 2>&1
if not errorlevel 1 set "PY_CMD=py -3.12"
if defined PY_CMD goto :python_found

python "%HELPER%" check-python --quiet >nul 2>&1
if not errorlevel 1 set "PY_CMD=python"
if defined PY_CMD goto :python_found

goto :python_missing

:python_found
%PY_CMD% "%HELPER%" check-windows
if errorlevel 2 goto :unsupported_windows
echo.
echo Using this Python for the project:
%PY_CMD% "%HELPER%" check-python
goto :step_cmake

:unsupported_windows
echo.
echo SETUP STOPPED - this 32-bit Windows version is not supported.
echo 64-bit Windows 10 or 11 is required, because 64-bit Python and
echo the Microsoft C++ build tools are 64-bit only.
goto :fail

:python_missing
echo Python was not found on this PC.
echo.
echo ------------------------------------------------------------
echo Missing prerequisite: Python 3.10
echo ------------------------------------------------------------
echo Software      : Python 3.10 - winget package Python.Python.3.10
echo Why IAMS needs it:
echo   IAMS is written in Python. Python 3.10 is the version this
echo   project is verified on. Python 3.11 and 3.12 also satisfy the
echo   dependency constraints, while 3.13 and newer are not supported.
echo Status        : REQUIRED - setup cannot continue without it.
echo Download size : about 30 MB.
echo Permissions   : winget asks for administrator approval only if it
echo                 is really needed. Python is normally installed
echo                 for the current user.
echo Restart       : normally not needed. This window does not see a
echo                 changed PATH, so setup re-checks Python itself.
echo.
where winget >nul 2>&1
if errorlevel 1 goto :python_no_winget
set "ANS="
set /p ANS="Would you like the setup script to install this prerequisite now? (Y/N) "
if /i "%ANS%"=="Y" goto :install_python
goto :python_declined

:install_python
echo.
echo Installing Python 3.10 with winget. Please wait ...
winget install --id Python.Python.3.10 --exact --accept-package-agreements --accept-source-agreements
set "IAMS_RC=%ERRORLEVEL%"
echo.
echo Re-checking whether Python is now available ...
py -3.10 "%HELPER%" check-python --quiet >nul 2>&1
if not errorlevel 1 set "PY_CMD=py -3.10"
if defined PY_CMD goto :python_installed
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" "%HELPER%" check-python --quiet >nul 2>&1
if not errorlevel 1 set "PY_CMD="%LOCALAPPDATA%\Programs\Python\Python310\python.exe""
if defined PY_CMD goto :python_installed
if exist "%ProgramFiles%\Python310\python.exe" "%ProgramFiles%\Python310\python.exe" "%HELPER%" check-python --quiet >nul 2>&1
if not errorlevel 1 set "PY_CMD="%ProgramFiles%\Python310\python.exe""
if defined PY_CMD goto :python_installed
goto :python_install_unverified

:python_installed
echo Python is now available.
%PY_CMD% "%HELPER%" check-windows
if errorlevel 2 goto :unsupported_windows
echo.
echo Using this Python for the project:
%PY_CMD% "%HELPER%" check-python
goto :step_cmake

:python_install_unverified
echo.
echo SETUP STOPPED - Python could not be confirmed as installed.
echo The winget command returned exit code %IAMS_RC%.
echo.
echo What to do next:
echo   1. If an installation window asked you to restart, restart
echo      Windows first.
echo   2. Close this window, open a NEW Command Prompt in the project
echo      folder and run setup.bat again. A new window is needed
echo      because PATH changes only affect new processes.
echo   3. If winget could not install Python, install it manually:
echo         https://www.python.org/downloads/windows/
echo      Choose Python 3.10 64-bit and tick "Add Python to PATH".
echo.
echo Nothing was changed in this project - it is safe to run again.
goto :action_required

:python_declined
echo.
echo You chose not to install Python.
echo.
echo How to install it manually:
echo   1. Open the official download page:
echo         https://www.python.org/downloads/windows/
echo   2. Download the Python 3.10 64-bit installer.
echo   3. Run it and tick "Add Python to PATH".
echo   4. Close this window, open a NEW Command Prompt in the project
echo      folder and run setup.bat again.
echo.
echo SETUP STOPPED - Python is required to continue.
goto :fail

:python_no_winget
echo winget is not available on this PC, so setup cannot install
echo Python for you.
goto :python_declined

REM =====================================================================
REM [2/7] CMake
REM =====================================================================
:step_cmake
echo.
echo ------------------------------------------------------------
echo [2/7] Checking CMake
echo ------------------------------------------------------------
echo.
set "BUILD_NEEDED=1"
if not exist ".venv\Scripts\python.exe" goto :build_needed
".venv\Scripts\python.exe" "%HELPER%" check-python --quiet >nul 2>&1
if errorlevel 1 goto :build_needed
".venv\Scripts\python.exe" "%HELPER%" check-dlib --quiet >nul 2>&1
if errorlevel 1 goto :build_needed
set "BUILD_NEEDED=0"
echo   dlib is already installed in the existing .venv, so no C++
echo   toolkit is required right now.
goto :cmake_detect

:build_needed
echo   dlib, the face recognition engine, is not built for this
echo   project yet. pip compiles it from source, and that build uses
echo   CMake and the Microsoft C++ compiler.

:cmake_detect
echo.
%PY_CMD% "%HELPER%" detect-cmake --env-file "%CMAKE_ENV%"
set "IAMS_CMAKE_OK="
set "IAMS_CMAKE_BIN="
if exist "%CMAKE_ENV%" call "%CMAKE_ENV%"
if "%IAMS_CMAKE_OK%"=="1" goto :cmake_ready
goto :cmake_missing

:cmake_ready
if not defined IAMS_CMAKE_BIN goto :step_msvc
echo.
echo Adding this folder to PATH for the setup session only:
echo   %IAMS_CMAKE_BIN%
set "PATH=%IAMS_CMAKE_BIN%;%PATH%"
goto :step_msvc

:cmake_missing
if "%BUILD_NEEDED%"=="0" goto :cmake_not_needed
echo.
echo ------------------------------------------------------------
echo Missing prerequisite: CMake
echo ------------------------------------------------------------
echo Software      : CMake - winget package Kitware.CMake
echo Why IAMS needs it:
echo   pip builds dlib from source, because dlib has no prebuilt
echo   Windows package. That build uses CMake to generate the
echo   Visual Studio project files.
echo Status        : REQUIRED while dlib has to be built.
echo Download size : about 100 MB.
echo Permissions   : winget asks for administrator approval only if
echo                 it is really needed.
echo Restart       : normally not needed. setup re-checks CMake by its
echo                 full path instead of relying on a new PATH.
echo.
where winget >nul 2>&1
if errorlevel 1 goto :cmake_no_winget
set "ANS="
set /p ANS="Would you like the setup script to install this prerequisite now? (Y/N) "
if /i "%ANS%"=="Y" goto :install_cmake
goto :cmake_declined

:install_cmake
echo.
echo Installing CMake with winget. Please wait ...
winget install --id Kitware.CMake --exact --accept-package-agreements --accept-source-agreements
set "IAMS_RC=%ERRORLEVEL%"
echo.
echo Re-checking CMake ...
%PY_CMD% "%HELPER%" detect-cmake --quiet --env-file "%CMAKE_ENV%" >nul 2>&1
set "IAMS_CMAKE_OK="
set "IAMS_CMAKE_BIN="
if exist "%CMAKE_ENV%" call "%CMAKE_ENV%"
if "%IAMS_CMAKE_OK%"=="1" goto :cmake_installed
goto :cmake_install_failed

:cmake_installed
echo CMake is now available.
if not defined IAMS_CMAKE_BIN goto :step_msvc
set "PATH=%IAMS_CMAKE_BIN%;%PATH%"
goto :step_msvc

:cmake_install_failed
echo.
echo SETUP STOPPED - CMake could not be confirmed as installed.
echo The winget command returned exit code %IAMS_RC%.
echo.
echo What to do next:
echo   1. Read the winget output above. A cancelled administrator
echo      prompt or a failed download are common causes.
echo   2. Install CMake manually from the official page:
echo         https://cmake.org/download/
echo      Choose the Windows x64 Installer and let it add CMake to
echo      the system PATH.
echo   3. Close this window, open a NEW Command Prompt in the project
echo      folder and run setup.bat again.
echo.
echo Nothing was changed in this project - it is safe to run again.
goto :action_required

:cmake_declined
echo.
echo You chose not to install CMake.
echo.
echo How to install it manually:
echo   1. Open the official download page: https://cmake.org/download/
echo   2. Download the Windows x64 Installer of the latest CMake.
echo   3. During installation choose "Add CMake to the system PATH".
echo   4. Close this window, open a NEW Command Prompt in the project
echo      folder and run setup.bat again.
echo.
echo SETUP STOPPED - CMake is required to build dlib on this PC.
goto :fail

:cmake_no_winget
echo winget is not available on this PC, so setup cannot install
echo CMake for you.
goto :cmake_declined

:cmake_not_needed
echo CMake was not found. It is not required right now, because dlib
echo is already available in the project environment.
echo.
echo If dlib ever has to be compiled again, install CMake from
echo   https://cmake.org/download/
goto :step_msvc

REM =====================================================================
REM [3/7] Microsoft C++ build tools
REM =====================================================================
:step_msvc
echo.
echo ------------------------------------------------------------
echo [3/7] Checking the Microsoft C++ build tools
echo ------------------------------------------------------------
echo.
echo Checking the C++ compiler, the linker and the Windows SDK ...
echo.
%PY_CMD% "%HELPER%" detect-msvc --env-file "%MSVC_ENV%"
set "IAMS_MSVC_OK="
set "IAMS_VS_HAS_CPP="
set "IAMS_VS_INSTANCE="
set "IAMS_VS_INSTALLER="
if exist "%MSVC_ENV%" call "%MSVC_ENV%"
if "%IAMS_MSVC_OK%"=="1" goto :msvc_ready
goto :msvc_missing

:msvc_ready
echo.
echo These tools are complete, so dlib can be compiled if needed.
goto :step_venv

:msvc_missing
if "%BUILD_NEEDED%"=="0" goto :msvc_not_needed
set "HAVE_WINGET="
where winget >nul 2>&1
if not errorlevel 1 set "HAVE_WINGET=1"
if defined IAMS_VS_INSTANCE goto :msvc_prompt
if not defined HAVE_WINGET goto :msvc_no_winget

:msvc_prompt
echo.
echo ------------------------------------------------------------
echo Missing prerequisite: Microsoft C++ build tools
echo ------------------------------------------------------------
echo Why IAMS needs it:
echo   Compiling dlib from source needs the MSVC C++ compiler
echo   cl.exe, the linker link.exe and the Windows SDK headers and
echo   libraries. Installing a Build Tools product alone is not
echo   enough - the workload "Desktop development with C++" and its
echo   Windows SDK components must be selected as well.
echo Status        : REQUIRED while dlib has to be built.
echo Download size : LARGE - roughly 3 to 7 GB.
echo Time          : typically 10 to 30 minutes.
echo Permissions   : administrator approval is required. The official
echo                 installer asks for it - setup never bypasses UAC.
echo Restart       : a restart may be required. If so, setup tells you
echo                 exactly what to do next.
if defined IAMS_VS_INSTANCE echo Method        : add the workload "Desktop development with C++"
if defined IAMS_VS_INSTANCE echo                 to the Visual Studio already installed at
if defined IAMS_VS_INSTANCE echo                   %IAMS_VS_INSTANCE%
if not defined IAMS_VS_INSTANCE echo Method        : install the Microsoft Visual Studio 2022
if not defined IAMS_VS_INSTANCE echo                 Build Tools through winget
echo.
set "ANS="
set /p ANS="Would you like the setup script to install this prerequisite now? (Y/N) "
if /i not "%ANS%"=="Y" goto :msvc_declined
if defined IAMS_VS_INSTANCE goto :msvc_modify
goto :msvc_winget

:msvc_modify
echo.
echo Adding the C++ workload with the Visual Studio Installer.
echo This can take 10 to 30 minutes, and Windows will ask for
echo administrator approval. Please wait ...
echo.
"%IAMS_VS_INSTALLER%" modify --installPath "%IAMS_VS_INSTANCE%" --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --includeRecommended --passive --norestart
set "IAMS_RC=%ERRORLEVEL%"
goto :msvc_recheck

:msvc_winget
echo.
echo Installing the Microsoft Visual Studio 2022 Build Tools with
echo winget, including the workload "Desktop development with C++"
echo and a Windows SDK. This is a large download - please wait ...
echo.
winget install --id Microsoft.VisualStudio.2022.BuildTools --exact --accept-package-agreements --accept-source-agreements --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --includeRecommended"
set "IAMS_RC=%ERRORLEVEL%"
goto :msvc_recheck

:msvc_recheck
echo.
echo Re-checking the C++ compiler, the linker and the Windows SDK ...
%PY_CMD% "%HELPER%" detect-msvc --quiet --env-file "%MSVC_ENV%" >nul 2>&1
set "IAMS_MSVC_OK="
set "IAMS_VS_HAS_CPP="
set "IAMS_VS_INSTANCE="
set "IAMS_VS_INSTALLER="
if exist "%MSVC_ENV%" call "%MSVC_ENV%"
if "%IAMS_MSVC_OK%"=="1" goto :msvc_installed
goto :msvc_install_unverified

:msvc_installed
echo The Microsoft C++ build tools are now available.
goto :step_venv

:msvc_install_unverified
echo.
echo SETUP STOPPED - the C++ build tools could not be confirmed.
echo The installer returned exit code %IAMS_RC%.
if "%IAMS_RC%"=="3010" goto :msvc_reboot
if "%IAMS_RC%"=="1641" goto :msvc_reboot
echo.
echo What to do next:
echo   1. If the Visual Studio Installer window is still open, let it
echo      finish, then close it.
echo   2. Check the reason above. A cancelled administrator prompt or
echo      a failed download are common causes.
echo   3. You can also install the workload manually:
echo         https://learn.microsoft.com/en-us/cpp/overview/acquire-msvc
echo      In the installer select the workload "Desktop development
echo      with C++" and make sure "MSVC v143 build tools" and a
echo      "Windows 11 SDK" are ticked in the installation details.
echo   4. Close this window, open a NEW Command Prompt in the project
echo      folder and run setup.bat again.
echo.
echo Nothing was changed in this project - it is safe to run again.
goto :action_required

:msvc_reboot
echo.
echo Windows reported that a restart is required before the new
echo build tools can be used.
echo.
echo What to do next:
echo   1. Restart Windows.
echo   2. Open a NEW Command Prompt in the project folder.
echo   3. Run setup.bat again - it continues from this point.
goto :action_required

:msvc_declined
echo.
echo You chose not to install the Microsoft C++ build tools.
echo.
echo How to install them manually:
echo   1. Open the official documentation:
echo        https://learn.microsoft.com/en-us/cpp/overview/acquire-msvc
echo   2. Download "Build Tools for Visual Studio" and run it.
echo   3. Select the workload "Desktop development with C++" and make
echo      sure these are ticked in the installation details:
echo        - MSVC v143 build tools
echo        - Windows 11 SDK
echo   4. Restart the PC if Windows asks for it.
echo   5. Open a NEW Command Prompt in the project folder and run
echo      setup.bat again.
echo.
echo SETUP STOPPED - the C++ build tools are required to build dlib
echo on this PC.
goto :fail

:msvc_no_winget
echo winget is not available on this PC, so setup cannot install the
echo Microsoft C++ build tools for you.
goto :msvc_declined

:msvc_not_needed
echo The Microsoft C++ build tools were not found. They are not
echo required right now, because dlib is already available in the
echo project environment.
echo.
echo If dlib ever has to be compiled again, install them from
echo   https://learn.microsoft.com/en-us/cpp/overview/acquire-msvc
goto :step_venv

REM =====================================================================
REM [4/7] Virtual environment
REM =====================================================================
:step_venv
echo.
echo ------------------------------------------------------------
echo [4/7] Preparing the virtual environment .venv
echo ------------------------------------------------------------
echo.

if exist ".venv\Scripts\python.exe" goto :venv_exists
echo Creating a new virtual environment with %PY_CMD% ...
%PY_CMD% -m venv .venv
if errorlevel 1 goto :venv_create_failed
echo Virtual environment created.
goto :venv_ready

:venv_exists
echo Found an existing virtual environment - validating it ...
".venv\Scripts\python.exe" "%HELPER%" check-venv
if errorlevel 1 goto :venv_broken
echo The existing virtual environment is usable and will be reused.
goto :venv_ready

:venv_create_failed
echo.
echo SETUP STOPPED - the virtual environment could not be created.
echo Check that this interpreter works: %PY_CMD% --version
echo Then run setup.bat again.
goto :fail

:venv_broken
echo.
echo The existing .venv is broken (see the reason above), for example
echo damaged package files after an interrupted install, or it was
echo created with an unsupported Python version such as 3.14.
echo.
echo Safe recovery - remove ONLY the virtual environment and let
echo setup.bat recreate it:
echo     rmdir /s /q .venv
echo This deletes .venv and nothing else. Databases, face encodings,
echo configuration and student data are not touched.
echo.
set "ANS="
set /p ANS="Remove .venv now and recreate it? (Y/N) "
if /i not "%ANS%"=="Y" goto :venv_broken_declined
rmdir /s /q .venv
if exist ".venv" goto :venv_remove_failed
echo Removed. Recreating ...
%PY_CMD% -m venv .venv
if errorlevel 1 goto :venv_create_failed
goto :venv_ready

:venv_broken_declined
echo.
echo You chose to keep the existing .venv.
echo.
echo SETUP STOPPED - a working .venv is required.
echo When you are ready, remove it with:  rmdir /s /q .venv
echo then run setup.bat again.
goto :fail

:venv_remove_failed
echo.
echo SETUP STOPPED - .venv could not be removed.
echo Close any program that uses it, run "rmdir /s /q .venv"
echo manually, then run setup.bat again.
goto :fail

:venv_ready

REM =====================================================================
REM [5/7] Python packages
REM =====================================================================
:step_packages
echo.
echo ------------------------------------------------------------
echo [5/7] Installing the Python packages
echo ------------------------------------------------------------
echo.
echo Checking whether the environment is already complete ...
".venv\Scripts\python.exe" "%HELPER%" verify-install --quiet >nul 2>&1
if not errorlevel 1 goto :packages_ready
echo Some of the required packages are still missing in .venv.
echo.
echo Packages      : the runtime dependencies from requirements.txt,
echo                 for example streamlit, opencv, mediapipe, numpy,
echo                 pandas, Pillow, fpdf2 and face-recognition.
echo                 dlib is compiled from source here - that is the
echo                 slow part and it needs the C++ build tools.
echo Why IAMS needs them:
echo                 they are the libraries the application imports
echo                 when it starts.
echo Status        : REQUIRED to run IAMS.
echo Permissions   : none - they are installed into the project
echo                 folder only, never system-wide.
echo Time          : a few minutes, longer when dlib is compiled.
echo.
set "ANS="
set /p ANS="Would you like the setup script to install the project packages now? (Y/N) "
if /i not "%ANS%"=="Y" goto :packages_declined
echo.
echo Installing the packages into .venv. Please wait ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 echo WARNING: pip could not be upgraded - continuing with the bundled pip.
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :install_failed
echo Packages installed.
goto :verify

:packages_ready
echo The environment already contains every critical package, so
echo nothing is installed again.
goto :verify

:packages_declined
echo.
echo You chose not to install the project packages.
echo.
echo Run setup.bat again when you are ready - it continues from this
echo point and installs the packages.
echo.
echo SETUP STOPPED - the packages are required to run IAMS.
goto :fail

:install_failed
echo.
echo SETUP FAILED - installing the packages failed. The reason is in
echo the output above. The most common causes:
echo   - dlib has no prebuilt Windows wheels and is compiled from
echo     source. This needs the Microsoft C++ build tools and CMake:
echo         winget install --id Microsoft.VisualStudio.2022.BuildTools
echo     and select the workload "Desktop development with C++".
echo   - No internet connection, or PyPI is blocked by a firewall.
echo   - An unsupported Python version - see README.md.
echo.
echo After fixing the cause, run setup.bat again.
goto :fail

REM =====================================================================
REM [6/7] Verification and [7/7] success message
REM =====================================================================
:verify
echo.
echo ------------------------------------------------------------
echo [6/7] Verifying the installation
echo ------------------------------------------------------------
echo.
".venv\Scripts\python.exe" "%HELPER%" verify-install
if errorlevel 1 goto :verify_failed

echo.
echo ------------------------------------------------------------
echo [7/7] Setup finished successfully
echo ------------------------------------------------------------
echo.
echo Launch IAMS with:
echo     run.bat
echo.
echo run.bat starts the Streamlit server and opens
echo http://localhost:8501 in your browser. A webcam is required
echo for the real-time attendance page.
echo.
pause
exit /b 0

:verify_failed
echo.
echo SETUP FAILED - some critical packages could not be imported.
echo See the missing packages above, then run setup.bat again.
goto :fail

:helper_missing
echo SETUP FAILED - scripts\setup_helper.py is missing.
echo The download or extraction is incomplete.
echo Re-extract the ZIP archive and run setup.bat again.
goto :fail

:action_required
echo.
echo Setup will continue from this same point when you run
echo setup.bat again.
echo.
pause
exit /b 2

:fail
echo.
pause
exit /b 1
