# Intelligent Attendance Monitoring System Using Computer Vision

## 📌 Overview

This project is an AI-powered attendance system that automates student attendance using Computer Vision and face recognition techniques. A camera recognizes each student, checks that the face is a real, live person (not a photo or a phone screen), and marks them present or late in the running session.

## 🚀 Features

**Attendance**
* Real-time face detection and recognition (one camera, one student at a time)
* Automatic marking as **present** or **late** (late threshold per session)
* Liveness check against cheating: random blinks + a random head turn, plus anti-spoof models that catch photos and phone / screen replays
* People far behind the student are ignored, so a queue does not block marking (two people side by side still have to step up one at a time)
* A small photo is saved with every mark, so an instructor can check it later

**Sessions**
* Create sessions as **Scheduled** or **Active (start now)**, with an optional end time
* Scheduled sessions **start by themselves** at their start time and **close by themselves** at their end time — no page needs to be open, and missed times are caught up after a restart
* Start a session by hand at any time. Started **after** its scheduled time, it starts fresh: lateness counts from the moment it was started
* Edit any session at any time; changing its start time or late threshold re-checks who was present or late
* Session details list **every** expected student — present, late and absent — with counts, a status filter and a search box
* **Fix attendance by hand**: change any student's status (Present / Late / Absent) right in that list — e.g. a student the camera missed. Changes are labelled *Instructor* and are kept even if the session's times are edited later
* The camera turns itself off when its session ends

**Students & reports**
* Student registry with face enrollment from the camera or from photos
* Bulk import of students (Excel / CSV) and face photos (ZIP), with a duplicate-face check
* Statistics with **absences and attendance rates** (per student, session and course), a "most absences" list, CSV export and PDF reports
* AI Assistant: ask questions about attendance in plain language (needs an API key)

## 🧠 Technologies Used

* Python 3.10 (see Python version note below)
* OpenCV (YuNet face detection, SFace face recognition, MiniFASNet anti-spoofing)
* MediaPipe (face landmarks for blink / head-turn liveness, phone/screen detection)
* face_recognition (dlib) - optional legacy recogniser, see `requirements-dlib.txt`
* NumPy & Pandas
* SQLite
* Streamlit (for UI)

## ⚠️ Python Version Compatibility

| Version | Status | Notes |
|---------|--------|-------|
| **3.10** | ✅ **Recommended, verified** | Tested end-to-end on this project's development machine |
| 3.11, 3.12 | 🟡 Constraint-checked, untested | Every pinned dependency publishes compatible Windows wheels, but this combination has not been run on a real machine |
| 3.13+ | ❌ Not supported | untested; the optional `face_recognition` (dlib) failed on 3.14, see `PHASE_2A_INSTALLATION_NOTES.txt` |

Use the 64-bit installer from [python.org](https://www.python.org/downloads/windows/) and tick **Add Python to PATH**. `setup.bat` probes for 3.10 first, then 3.11 and 3.12.

Every package in `requirements.txt` installs from a prebuilt wheel, so **no C++ compiler or CMake is needed**. Face recognition uses OpenCV's SFace model. The previous dlib recogniser is optional (`requirements-dlib.txt`, `setup.bat --with-dlib`); dlib publishes no prebuilt Windows wheels and must be compiled, and `face_recognition_models` needs `setuptools<81` for `pkg_resources`.

## 📦 Quick Start (Windows)

### Prerequisites

- Windows 10/11, 64-bit
- Python 3.10 (recommended) added to PATH — or let `setup.bat` install it for you after asking
- A webcam, and Windows camera permission enabled for desktop apps
- **Only for the optional dlib recogniser** (`setup.bat --with-dlib`): Microsoft C++ Build Tools + CMake, because dlib has **no prebuilt Windows wheels** and is compiled from source. Not needed for a normal installation. `setup.bat --with-dlib` checks for them and asks before installing:
  ```cmd
  winget install --id Kitware.CMake
  winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --includeRecommended"
  ```
  The Build Tools download is **large (roughly 3–7 GB), requires administrator approval (UAC) and may require a restart**; with the graphical installer select the **"Desktop development with C++"** workload.
- Extract the project to a **short local folder** (e.g. `C:\IAMS`). Avoid OneDrive-synced folders — background syncing can corrupt a virtual environment mid-install — and do not extract a ZIP into a folder of the same name: a long path can push files inside `.venv` past Windows' 260-character limit. `setup.bat` warns when the path is too long.

### Automated Setup & Launch

1. **Download & Extract** the project ZIP to a local folder.

2. **Run Setup** (first time only, or after changing dependencies):
   ```cmd
   setup.bat
   ```
   The script walks through seven steps and **asks before installing anything**:

   1. Windows / free-disk-space check (64-bit required)
   2. Python — probes for 3.10, then 3.11/3.12; offers `winget install Python.Python.3.10` if none is found
   3. CMake — skipped unless `setup.bat --with-dlib` (optional dlib recogniser)
   4. Microsoft C++ build tools — skipped unless `--with-dlib`; then it checks the real components (`cl.exe`, `link.exe`, a Windows SDK)
   5. `.venv` — created, or validated and reused; a broken `.venv` (wrong Python, or damaged package files after an interrupted install) is only removed after you confirm
   6. Runtime packages from `requirements.txt` (plus `requirements-dlib.txt` with `--with-dlib`) — skipped entirely when the environment is already healthy
   7. Final import check of every critical package

   For every missing prerequisite the script prints the software name, why IAMS needs it, whether it is required or optional, whether administrator approval / a large download / a restart may be needed, and then asks **`Would you like the setup script to install this prerequisite now? (Y/N)`**. It never bypasses UAC or installer consent, and it never installs silently.

   * Answer **N** → setup does not install it and stops with the official download link and manual instructions (required prerequisites only; optional ones are simply skipped).
   * Installation is verified by **re-detecting the actual components**, not by trusting the installer. If a prerequisite cannot be confirmed afterwards, setup says so and tells you what to do instead of reporting success.
   * If a **restart** or a **new terminal** is needed, setup stops cleanly — restart, open a new Command Prompt in the project folder, and run `setup.bat` again; it continues from that step.

   The first run downloads the packages and takes a few minutes (nothing is compiled; with `--with-dlib` compiling dlib adds 10–30 minutes); later runs take about a minute. Exit codes: `0` = success, `1` = stopped or failed, `2` = a restart or a new terminal is needed. The script is safe to re-run at any time and never touches databases, face encodings, or student data — those live in `data/` and are created on first launch.

3. **Launch Application**:
   ```cmd
   run.bat
   ```
   This starts the app with the `.venv` interpreter (nothing is reinstalled) and opens `http://localhost:8501` in your browser. It runs `run_app.py`, which first starts the **session scheduler** (the part that starts and closes scheduled sessions on time) and then the Streamlit server. Only **this PC** can open the app: other devices on the network (Wi-Fi) are refused, so nobody else can reach it and passwords never cross the network. Use from other devices comes later, together with HTTPS.

4. **Uninstall** (optional):
   ```cmd
   uninstall.bat
   ```
   Asks about each part separately, shows a summary, and changes nothing until you confirm:
   * **In the project folder:** `.venv` with every Python library, or only the optional dlib recogniser; the temporary files; your data (you have to type `DELETE`); or the whole IAMS folder.
   * **Shared with the rest of the PC:** the Microsoft C++ build tools (the Build Tools product, or only the C++ workload of a full Visual Studio), CMake, Python 3.10, and the pip / Streamlit / Matplotlib caches. `setup.bat` records the prerequisites it installs in `.iams_installed`, so the uninstaller tells you which ones were already on the PC and may be used by other programs.

   A running IAMS is closed first. Exit codes: `0` = finished, `1` = a part could not be removed, `2` = restart Windows to finish removing the C++ build tools.

### Manual Alternative
If you prefer manual control:
```cmd
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run_app.py
```
`run_app.py` starts the session scheduler (scheduled sessions start and close on time even when no page is open) and then Streamlit; it accepts the usual Streamlit options. A plain `streamlit run app.py` also works, but then the scheduler only starts once the first page is opened.

### Installing the software vs. your data
Setup only ever touches **software**: Python, CMake, the C++ Build Tools, the `.venv` folder and the packages inside it. It never creates or changes **your data** — the database (`data/database/`), face encodings (`data/face_encodings/`), enrollment photos (`data/enrollment_images/`), unknown-face captures (`data/unknown_faces/`), logs and `.env`. Moving the system to another PC therefore means moving `data/` and `.env` yourself; re-running `setup.bat` rebuilds the software but does **not** re-enroll students — enrollment happens in the app's Student Registry.

## 🎯 Using the System

### Signing in
* **First start:** the app asks you to create the first **IT account**. That account creates every other account on the **Users** page.
* **Roles:** *Instructor* < *Admin* < *IT*.

  | | Instructor | Admin | IT |
  |---|---|---|---|
  | Home, Real-Time Attendance, Session Management, Statistics & Reports | ✅ own classes | ✅ all | ✅ all |
  | My Classes (their sections: start a session, class list) | ✅ | if they teach | if they teach |
  | Student Registry, Courses & Sections, AI Assistant | ❌ | ✅ | ✅ |
  | Users: instructor accounts | ❌ | ✅ | ✅ |
  | Users: Admin and IT accounts | ❌ | ❌ | ✅ |

* **New accounts** get a temporary password, shown once. Its owner must choose their own password at the first sign-in. **Reset password** on the Users page works the same way.
* **Lockout:** 5 wrong passwords in a row lock the account for 15 minutes. An Admin / IT user can **Unlock** it sooner.
* **Passwords** are stored only as scrypt hashes, never as text. Change yours with **🔑 Password** in the sidebar.
* **Forgot your password?** Every account gets a one-time **recovery code**, shown once (after creating the first account, after choosing your own password, or with **🧾 New recovery code** in the sidebar). Save it somewhere safe. On the sign-in page, **Forgot your password?** takes your username + the code + a new password. The code is then used up and a new one is shown. Wrong codes count towards the same lockout as wrong passwords. No code? Instructors ask an Admin, Admins ask IT.
* **Keep two IT accounts:** while only one exists, IT users see a warning on Home and Users, because a lost IT password plus a lost code would leave nobody to manage Admin / IT accounts.
* **Instructors see only their own classes:** an Admin links each section to its instructor's account (Courses & Sections → section settings → **Instructor account**; a class-list import also links an `instructor` that matches an account's username or full name). An instructor then sees only the sessions of their sections and the ones they created: on Home, My Classes, Real-Time Attendance, Session Management and Statistics (*My Sessions*, *My Attendance History*). They create sessions only for their own sections, can't delete sessions, and "Created by" is always the signed-in person. A section without an account shows "(no account)".
* **Signing out:** **🚪 Log out**, or refresh / close the browser tab.
* **Last resort for technical staff** (password *and* recovery code lost, and no second IT account): on this PC run:

  ```bash
  .venv\Scripts\python scripts\manage_accounts.py reset-password <username>
  ```

  Also available: `list`, `unlock <username>` and `create-it <username> "<full name>"`.

### Pages
After signing in, the sidebar shows the pages your role may open:

1. **Home** - the start page, top to bottom:
   * **Getting started** - the setup steps (add students → add face photos → create a session → take attendance), ticked off automatically, with a **Go** button for the next one. It folds away once all are done.
   * **Today** - quick buttons (**▶️ Start a class**, new session, add students, reports, AI Assistant), a card for every running session with live counts (expected / present / late / absent / attendance %) and **Open camera** / **Details** buttons, today's sessions and the next scheduled one. It refreshes itself every 10 seconds.
   * **Overview** - active students, students without face data, attendance rate, sessions in the last 7 days, and a 14-day chart.
   * **System check** - face models, the session timer, recognition data, face data, camera on/off and the AI Assistant key. It never turns the camera on.
2. **Real-Time Attendance** - pick an **active** session, press **Start Attendance Stream**, and students in front of the camera are recognized, liveness-checked and marked. A scheduled session can be started early here with **Activate Session**. In a session that belongs to a **section**, only students on its class list are marked; anyone else is shown as **Not in this class**. Pick the webcam under **📷 Camera** in the sidebar (also shown before taking face photos). *Automatic* uses the first real camera and skips phones linked to the PC; the choice is remembered, and if the chosen camera is unplugged you get a clear message instead of another camera opening.
3. **Session Management** - three tabs:
   * **Sessions List** - every session with its status and present / late counts. Click a row to get its buttons: **Start Now**, **Close**, **View**, **Edit**, **Delete** (an active session must be closed before it can be deleted).
   * **Create New Session** - first pick **Course → Section** (or *No section*), then name, start date and time, optional end time, late threshold, and *Scheduled* or *Active (start now)*. The section can also be changed later with **Edit**.
   * **Session Details & Records** - the session's info, the counts (expected / present / late / absent / attendance %) and the list of students, sorted present → late → absent. **Arrived** shows the time present / late was judged by when it is earlier than *Marked At* (first seen by the camera, or the start of the line the student was in). Use the filter buttons or the search box to find someone; at most 500 rows are shown at once to keep the page fast. **To fix attendance**, double-click a student's *Status* and pick Present, Late or Absent — it is saved at once (possible once the session has started, also after it closed). In a section session, **Mark a student who is not on the class list** adds someone who attended but isn't on the list.
4. **Student Registry** - add / edit students, manage their face photos, bulk import.
5. **Courses & Sections** - three tabs:
   * **Courses** - add, edit, deactivate or delete courses (e.g. TM103). Select a row for its buttons.
   * **Sections & class lists** - pick a course, add its sections (e.g. A, B, Lab 1) with the instructor's name, and manage each section's **class list**: search and pick students, paste a list of student IDs, remove selected students, or download the list as CSV. **▶️ Start session now** starts a session for the section at once and opens the camera page (the camera itself still starts only when you press *Start Attendance Stream*).
   * **Import class lists** - an Excel / CSV file with the columns `course_code`, `section`, `student_id` (optional: `course_name`, `instructor`). Courses and sections that don't exist yet are created; you see a preview first. Importing the same file again is safe, and an import never removes anyone.
   * A course or section that has sessions can't be deleted - deactivate it instead, so its history is kept.
6. **Statistics & Reports** - four tabs: an overview (attendance rate, present / late / absent totals, charts, the students with the most absences), one student's record (search by name or ID; attended and missed sessions), one session or course (filter by course / section; session reports show the section and instructor; a **section summary** table), and a searchable history. Every tab has CSV / PDF downloads; the session CSV lists every expected student, absent ones included. Absence totals count **closed** sessions only, where attendance is final.
7. **AI Assistant** - ask questions about attendance in plain language (requires an OmniRoute API key in `.env`).
8. **Users** (Admin / IT) - create accounts, change a name or role, reset a password, unlock, deactivate.
9. **Activity Log** (Admin / IT) - who did what, and when: sign-ins (also failed ones and lockouts), accounts, students and face photos, courses and class lists, sessions, attendance changed by hand, check-outs and spot checks, the camera choice. Filter by dates, person, kind and text; click a row for its details (e.g. what an edit changed); **⬇️ Save as CSV**. Entries can't be edited or deleted. Background work (sessions / checks that start or end by themselves) shows as *System (automatic)*. The camera's own marks aren't listed (each record already says *Camera*), and passwords / recovery codes are never written. Each session's own history is also under **Session Details → 🕘 History of this session**.

### How sessions behave

| Situation | What happens |
|---|---|
| A scheduled session's start time arrives | It becomes **Active** within about 5 seconds, even if no one has the app open |
| An active session's end time arrives | It becomes **Closed**, and a running camera turns itself off |
| The app was off when those times passed | On the next start, sessions that should have started are started; sessions whose end time also passed are closed |
| A session is started by hand **after** its scheduled time | It starts fresh: its start time becomes "now", so nobody is late because the session began late |
| A session is started by hand **before** its scheduled time | It keeps its scheduled start time |
| The start time or late threshold of a session is edited | Every mark in it is re-checked as present or late against the new values |
| A session has no end time | It stays active until someone closes it |
| A student takes a few seconds for the blink / head-turn test | Present or late counts from the moment the camera **first recognised** them, not from the end of the test (if the test ends within 60 seconds; someone who leaves and comes back later counts from their return) |
| A line of students is still at the camera when the late time passes | Students who step up one after another (at most 20 seconds between them) form one line. A line that began **before** the late time keeps everyone in it **Present**, for up to 5 minutes after the late time. Their record says *In the line at the camera since …* |
| A session belongs to a **section** | Only the section's class list is expected and counted absent (students added to the list before the session started), plus anyone marked by hand. The camera marks only class-list students; it still recognises everyone, so a visitor is never mistaken for a look-alike on the list |
| A session has **check-out** on (setting when creating / editing it, or on *Start session now*) | In its last N minutes (default 10) the camera asks everyone to show their face again (**CHECK-OUT** on the video). Needs an end time to start by itself; **🚪 Start check-out now** starts it by hand |
| The instructor closes a session whose check-out hasn't run (e.g. no end time) | **⏹️ Close** asks first: **🚪 Check-out now, then close** runs the check-out and closes the session by itself when it ends, or **Close now, without check-out** |
| The instructor presses **🔎 Spot check now** (or schedules one at a time / **at a random time**, hidden until it starts) | For a few minutes (default 5) students who already checked in show their face again (**SPOT CHECK**), with the same liveness test |
| A check ends | Everyone present / late **before** it began but not seen during it becomes **🚪 Left early** (still counted as attended, shown in Session Details, statistics, CSV and PDF). Late arrivals aren't flagged for checks that ran before they came. A check the camera **never saw** (camera off the whole time) doesn't count, so nobody is flagged. With the camera on it counts even if nobody showed up |
| The instructor disagrees | In Session Details set the student's Status to Present / Late (clears Left early) or to 🚪 Left early |
| A session has **no section** (e.g. all sessions made before sections existed) | Every active student registered before it started is expected, as before |

**Who counts as "expected" (and so as absent if not marked)?** Every active student who was registered before the session started, plus anyone who was marked in it. Courses / class lists are planned for a later phase; then only that class's students will be expected.

Open pages notice when a session starts or closes by itself and refresh within about 10 seconds. Times are the attendance computer's local time.

### Adding many students at once (Student Registry → 📥 Bulk Import)

1. **Students** – upload an Excel (`.xlsx`) or CSV file with the columns `student_id`, `name` (required) and `email`, `department`, `notes` (optional); a template can be downloaded on the page. The file is checked first (new / updated / unchanged / rows with errors) and saved in one go. Re-uploading is safe: existing students are updated, never duplicated, and empty cells keep their current details.
2. **Face photos** – upload a ZIP of photos named after the student IDs (`2021001.jpg`, extra photos `2021001_2.jpg`, or a folder per student `2021001/…`). Each photo is checked (one clear face, big enough, not the same face as another student); problems are listed and downloadable as a report. Students who already have face data are skipped by default, so re-running the same ZIP continues an interrupted import.

## 📁 Project Structure

```
├── app.py                      # Streamlit entry point (pages and navigation)
├── run_app.py                  # Starts the session scheduler, then Streamlit (used by run.bat)
├── config.py                   # Centralized configuration
├── requirements.txt            # Runtime dependencies (pinned, verified)
├── requirements-dev.txt        # Test-only tooling (pytest)
├── setup.bat                   # Windows automated setup
├── run.bat                     # Windows automated launch
├── uninstall.bat               # Windows uninstaller - asks per part
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
│   ├── services/               # Business logic (attendance, enrollment, the session scheduler, etc.)
│   ├── realtime/               # Camera / vision engine process and its connection to the app
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
# A face narrower than this share of the face in front counts as someone in
# the background and is ignored (0 = any second face blocks marking)
# BACKGROUND_FACE_MAX_RATIO=0.5

# Present / late: count from the first time the camera recognised the student
# (if the liveness test ends within this many seconds)
# FIRST_SEEN_MAX_WAIT_SECONDS=60
# Line at the camera: at most this many seconds between two students
# LATE_LINE_GAP_SECONDS=20
# A line that began before the late time keeps students Present for this
# many minutes after it (0 = off)
# LATE_LINE_GRACE_MINUTES=5

# Server
# STREAMLIT_SERVER_PORT=8502
```

## 🧪 Running Tests

```cmd
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest tests/ -v
```

## 🐛 Troubleshooting

### Package installation fails with `No such file or directory` or `legacy-install-failure`
- **Cause:** the project folder path is too long (for example a ZIP extracted into a folder of the same name), so files inside `.venv` exceed Windows' 260-character path limit
- **Fix:** move the project to a short folder such as `C:\IAMS`, delete `.venv` there, and run `setup.bat` again

### `dlib` build fails during `setup.bat --with-dlib`
- **Note:** only the optional dlib recogniser needs this; a normal `setup.bat` does not compile anything
- **Cause:** dlib has no prebuilt Windows wheels, so pip compiles it from source — this requires the Microsoft C++ Build Tools and CMake
- **Fix:** re-run `setup.bat`; it detects the missing CMake / C++ components and offers to install them for you. `setup.bat` uses only official methods:
  ```cmd
  winget install --id Kitware.CMake
  winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --includeRecommended"
  ```
  In the Build Tools graphical installer, add the **"Desktop development with C++"** workload (with an MSVC v143 toolset and a Windows 11 SDK), then re-run `setup.bat`. Make sure CMake is on PATH or available to pip.
- **Note:** installing CMake or the Build Tools may require a **restart**. If so, `setup.bat` stops and tells you; restart, open a **new** terminal and run `setup.bat` again — it resumes from there.

### `No module named pkg_resources` (only with the optional dlib recogniser)
- **Cause:** the virtual environment contains setuptools 81+ (which removed `pkg_resources`) or no setuptools at all
- **Fix:** re-run `setup.bat --with-dlib` — it installs `setuptools>=65.5.0,<81` from `requirements-dlib.txt`. Manually: `.venv\Scripts\python.exe -m pip install "setuptools>=65.5.0,<81"`

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

### A scheduled session did not start
- **Check:** the app must be running at the start time (the scheduler runs inside it). If it was off, the session starts as soon as the app starts again — unless its end time has also passed, then it is closed.
- **Check:** the start time is the attendance computer's local time.

### "Multiple faces detected" although only one student is in front
- **Cause:** a second face of similar size is in view (e.g. someone standing right beside the student). Faces clearly further back are ignored.
- **Fix:** one student at a time in front of the camera; point the camera away from the queue. `BACKGROUND_FACE_MAX_RATIO` in `.env` sets how much smaller a face must be to count as background.

### "No virtual environment found" from `run.bat`
- **Fix:** run `setup.bat` first

### Port 8501 already in use
- **Fix:** stop other Streamlit instances, or set `STREAMLIT_SERVER_PORT=8502` in `.env`

### Installation fails behind a corporate firewall
- **Cause:** pip cannot reach PyPI to download packages, including the dlib source archive
- **Fix:** allow `pypi.org` and `files.pythonhosted.org`, or configure an internal mirror with `pip config set global.index-url <url>`

## 🗺️ Next Phases

Planned, not built yet:

* **Instructor screen** - an instructor picks their course and session and starts it
* **Use from several devices** - a school server with classroom PCs, HTTPS and device keys (today the app only opens on its own PC)
* **AI Assistant updates** - e.g. answering questions about absent students
* **New design** - a visual redesign of the whole app

## 📚 Documentation

* `PHASE_2A_INSTALLATION_NOTES.txt` - Detailed CV dependency installation notes, including the Python 3.14 failure
* `PHASE_2A_IMPLEMENTATION_REPORT.txt` - Phase 2A architecture details
* `REALTIME_ATTENDANCE_BUGS_FIX_REPORT.md` - Recent bug fixes
* `CAMERA_SMOOTHNESS_FIX_REPORT.md` - Performance optimizations

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 👨‍💻 Author

Abdelrahman Mohamed Saadallah
