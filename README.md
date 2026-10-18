# Intelligent Attendance Monitoring System Using Computer Vision

## 📌 Overview

This project is an AI-powered attendance system that automates student attendance using Computer Vision and face recognition techniques. The system detects and recognizes faces in real-time, records attendance, and tracks student presence duration during lectures.

## 🚀 Features

* Real-time face detection and recognition
* Automatic attendance recording
* Attendance duration tracking
* Late arrival and early leaving detection
* Liveness detection to prevent cheating
* Manual identity verification using student ID
* Adaptive face data updating for improved accuracy
* Instructor dashboard for monitoring and editing attendance
* Attendance reports generation

## 🧠 Technologies Used

* Python 3.10 (see Python version note below)
* OpenCV
* face_recognition (dlib-based)
* MediaPipe
* NumPy & Pandas
* SQLite
* Streamlit (for UI)

## ⚠️ Python Version Compatibility

| Version | Status | Notes |
|---------|--------|-------|
| **3.10** | ✅ **Recommended, verified** | Tested end-to-end on this project's development machine |
| 3.11, 3.12 | 🟡 Constraint-checked, untested | Every pinned dependency publishes compatible Windows wheels, but this combination has not been run on a real machine |
| 3.13+ | ❌ Not supported | `face_recognition` installation/imports fail; see `PHASE_2A_INSTALLATION_NOTES.txt` for the 3.14 failure details |

Use the 64-bit installer from [python.org](https://www.python.org/downloads/windows/) and tick **Add Python to PATH**. `setup.bat` probes for 3.10 first, then 3.11 and 3.12.

Two facts worth knowing:

* `face_recognition_models` imports `pkg_resources` at runtime, which **setuptools 81+ removed**. `requirements.txt` therefore pins `setuptools>=65.5.0,<81`; Python 3.12+ virtual environments ship without setuptools at all, so this pin is what makes the import work there.
* dlib publishes **no prebuilt Windows wheels** — it compiles from source during setup (see prerequisites below).

## 📦 Quick Start (Windows)

### Prerequisites

- Windows 10/11, 64-bit
- Python 3.10 (recommended) added to PATH — or let `setup.bat` install it for you after asking
- A webcam, and Windows camera permission enabled for desktop apps
- **Microsoft C++ Build Tools + CMake** — required because dlib has **no prebuilt Windows wheels** and is compiled from source. They are only needed while the environment is being built, and `setup.bat` asks before installing them:
  ```cmd
  winget install --id Kitware.CMake
  winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --includeRecommended"
  ```
  The Build Tools download is **large (roughly 3–7 GB, 10–30 minutes), requires administrator approval (UAC) and may require a restart**. If you install them by hand with the graphical installer, select the **"Desktop development with C++"** workload: installing the Build Tools *product* without that workload is not enough, because dlib needs the MSVC compiler (`cl.exe`), the linker (`link.exe`) and a Windows SDK. `setup.bat` verifies those components specifically, not just the presence of the product.
- Extract the project to a **local folder** (e.g. `C:\Projects\attendance-system`). Avoid OneDrive-synced folders — background syncing can corrupt a virtual environment mid-install.

### Automated Setup & Launch

1. **Download & Extract** the project ZIP to a local folder.

2. **Run Setup** (first time only, or after changing dependencies):
   ```cmd
   setup.bat
   ```
   The script walks through seven steps and **asks before installing anything**:

   1. Windows / free-disk-space check (64-bit required)
   2. Python — probes for 3.10, then 3.11/3.12; offers `winget install Python.Python.3.10` if none is found
   3. CMake — checked on PATH, in the standard folders and in the copy shipped with Visual Studio
   4. Microsoft C++ build tools — checks the real components (`cl.exe`, `link.exe`, a Windows SDK), not just whether a Build Tools product exists
   5. `.venv` — created, or validated and reused; a broken `.venv` (e.g. built with Python 3.14) is only removed after you confirm
   6. Runtime packages from `requirements.txt` — skipped entirely when the environment is already healthy
   7. Final import check of every critical package, including `face_recognition` and `pkg_resources`

   For every missing prerequisite the script prints the software name, why IAMS needs it, whether it is required or optional, whether administrator approval / a large download / a restart may be needed, and then asks **`Would you like the setup script to install this prerequisite now? (Y/N)`**. It never bypasses UAC or installer consent, and it never installs silently.

   * Answer **N** → setup does not install it and stops with the official download link and manual instructions (required prerequisites only; optional ones are simply skipped).
   * Installation is verified by **re-detecting the actual components**, not by trusting the installer. If a prerequisite cannot be confirmed afterwards, setup says so and tells you what to do instead of reporting success.
   * If a **restart** or a **new terminal** is needed, setup stops cleanly — restart, open a new Command Prompt in the project folder, and run `setup.bat` again; it continues from that step.

   The first run compiles dlib from source and can take **10–30 minutes** depending on the machine; later runs take about a minute. Exit codes: `0` = success, `1` = stopped or failed, `2` = a restart or a new terminal is needed. The script is safe to re-run at any time and never touches databases, face encodings, or student data — those live in `data/` and are created on first launch.

3. **Launch Application**:
   ```cmd
   run.bat
   ```
   This starts the Streamlit server with the `.venv` interpreter (nothing is reinstalled) and opens `http://localhost:8501` in your browser.

### Manual Alternative
If you prefer manual control:
```cmd
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

### Installing the software vs. your data
Setup only ever touches **software**: Python, CMake, the C++ Build Tools, the `.venv` folder and the packages inside it. It never creates or changes **your data** — the database (`data/database/`), face encodings (`data/face_encodings/`), enrollment photos (`data/enrollment_images/`), unknown-face captures (`data/unknown_faces/`), logs and `.env`. Moving the system to another PC therefore means moving `data/` and `.env` yourself; re-running `setup.bat` rebuilds the software but does **not** re-enroll students — enrollment happens in the app's Student Registry.

## 🎯 Using the System

1. **Student Registry** - Add/edit students via the sidebar navigation
2. **Session Management** - Create and activate attendance sessions
3. **Real-Time Attendance** - Start camera feed, students are auto-recognized
4. **Statistics & Reports** - View attendance analytics and export PDFs
5. **AI Assistant** - Query attendance data via natural language (requires OmniRoute API key in `.env`)

## 📁 Project Structure

```
├── app.py                      # Streamlit entry point
├── config.py                   # Centralized configuration
├── requirements.txt            # Runtime dependencies (pinned, verified)
├── requirements-dev.txt        # Test-only tooling (pytest)
├── setup.bat                   # Windows automated setup
├── run.bat                     # Windows automated launch
├── scripts/
│   └── setup_helper.py         # Python/dependency checks used by setup.bat
├── .env.example                # Environment variable template
├── .gitignore                  # Git ignore rules
├── data/                       # Local data (git-ignored, created on first run)
│   ├── database/               # SQLite database
│   ├── face_encodings/         # Student face embeddings (.npy)
│   ├── enrollment_images/      # Enrollment photos
│   ├── unknown_faces/          # Unrecognized face captures
│   └── logs/                   # Application logs
├── src/
│   ├── computer_vision/        # Face detection, embedding, recognition, liveness
│   ├── data/                   # Database schema, repositories
│   ├── services/               # Business logic (attendance, enrollment, etc.)
│   ├── ui/pages/               # Streamlit page components
│   ├── utils/                  # Logging, performance timing
│   └── ai/                     # LLM integration
├── tests/                      # Unit and integration tests
└── manual_tests/               # Manual testing scripts
```

The database, face encodings, enrollment images, and `.env` file are **not** included in the repository — they are local state, created or edited at runtime.

## ⚙️ Configuration

Copy `.env.example` to `.env` and adjust. The application starts without a `.env` file; it is only needed for the AI Assistant and optional overrides:

```ini
# Required for AI Assistant
OMNIROUTE_API_KEY=your_api_key_here

# Optional overrides
OMNIROUTE_BASE_URL=http://localhost:20128/v1
OMNIROUTE_MODEL=kr/claude-sonnet-4.5

# Application settings
# APP_DEBUG=false
# APP_LOG_LEVEL=INFO

# Computer Vision tuning
# FACE_DETECTION_CONFIDENCE=0.7
# FACE_RECOGNITION_THRESHOLD=0.6
# FACE_RECOGNITION_MARGIN=0.1
# LIVENESS_HEAD_TURN_ANGLE=15

# Server
# STREAMLIT_SERVER_PORT=8502
```

## 🧪 Running Tests

```cmd
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest tests/ -v
```

## 🐛 Troubleshooting

### `dlib` build fails during `setup.bat`
- **Cause:** dlib has no prebuilt Windows wheels, so pip compiles it from source — this requires the Microsoft C++ Build Tools and CMake
- **Fix:** re-run `setup.bat`; it detects the missing CMake / C++ components and offers to install them for you. `setup.bat` uses only official methods:
  ```cmd
  winget install --id Kitware.CMake
  winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --includeRecommended"
  ```
  In the Build Tools graphical installer, add the **"Desktop development with C++"** workload (with an MSVC v143 toolset and a Windows 11 SDK), then re-run `setup.bat`. Make sure CMake is on PATH or available to pip.
- **Note:** installing CMake or the Build Tools may require a **restart**. If so, `setup.bat` stops and tells you; restart, open a **new** terminal and run `setup.bat` again — it resumes from there.

### `No module named pkg_resources`
- **Cause:** the virtual environment contains setuptools 81+ (which removed `pkg_resources`) or no setuptools at all
- **Fix:** re-run `setup.bat` — it installs `setuptools>=65.5.0,<81` from `requirements.txt`. Manually: `.venv\Scripts\python.exe -m pip install "setuptools>=65.5.0,<81"`

### `setup.bat` reports an unsupported Python version
- **Cause:** only Python 3.13+ was found on the machine
- **Fix:** install Python 3.10 (64-bit, "Add to PATH" ticked), open a **new** terminal, re-run `setup.bat`

### `winget` is not available
- **Cause:** the Windows **App Installer** (`winget`) is missing, so `setup.bat` cannot install a missing prerequisite for you
- **Fix:** `setup.bat` prints the official link for the missing prerequisite and stops. Install it manually, then run `setup.bat` again:
  - Python 3.10 — <https://www.python.org/downloads/windows/> (tick "Add Python to PATH")
  - CMake — <https://cmake.org/download/> (Windows x64 installer, "Add CMake to the system PATH")
  - C++ Build Tools — <https://learn.microsoft.com/en-us/cpp/overview/acquire-msvc> (select the **"Desktop development with C++"** workload)

### The `.venv` is broken (e.g. created with the wrong Python)
- **Symptom:** `setup.bat` says the existing virtual environment is invalid
- **Fix:** let `setup.bat` remove it when asked, or manually:
  ```cmd
  rmdir /s /q .venv
  ```
  This deletes **only** the virtual environment — databases, face encodings, configuration and student data in `data/` are untouched. Then re-run `setup.bat`.

### Camera not detected
- **Fix:** ensure the webcam is connected and not used by another app, then allow camera access: *Windows Settings → Privacy & security → Camera → let desktop apps access your camera*

### "No virtual environment found" from `run.bat`
- **Fix:** run `setup.bat` first

### Port 8501 already in use
- **Fix:** stop other Streamlit instances, or set `STREAMLIT_SERVER_PORT=8502` in `.env`

### Installation fails behind a corporate firewall
- **Cause:** pip cannot reach PyPI to download packages, including the dlib source archive
- **Fix:** allow `pypi.org` and `files.pythonhosted.org`, or configure an internal mirror with `pip config set global.index-url <url>`

## 📚 Documentation

* `PHASE_2A_INSTALLATION_NOTES.txt` - Detailed CV dependency installation notes, including the Python 3.14 failure
* `PHASE_2A_IMPLEMENTATION_REPORT.txt` - Phase 2A architecture details
* `REALTIME_ATTENDANCE_BUGS_FIX_REPORT.md` - Recent bug fixes
* `CAMERA_SMOOTHNESS_FIX_REPORT.md` - Performance optimizations

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 👨‍💻 Author

Abdelrahman Mohamed Saadallah
