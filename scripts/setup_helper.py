#!/usr/bin/env python3
"""Environment checks for setup.bat / run.bat (standard library only).

This helper exists so that setup.bat never has to parse tool output,
version strings or inline multi-line ``python -c`` snippets. It is run by
candidate interpreters *before* any project dependency is installed, so it
must only import from the standard library.

Commands
--------
check-python        Exit 0 when THIS interpreter is supported by the
                    project (Python 3.10 / 3.11 / 3.12), exit 1 otherwise.
                    Pass --quiet to suppress the success line (used while
                    probing several interpreters).

check-venv          Run by the project .venv interpreter: exit 0 when it
                    is a supported Python AND no installed package is
                    damaged (a *.dist-info folder without its METADATA
                    file, left by an interrupted install - pip then fails
                    with "No such file or directory: ...METADATA").
                    Exit 1 otherwise, printing the reason.

check-windows       Report the Windows version, the CPU architecture and the
                    free disk space of the project drive.
                    Exit 0 = fine, 1 = warning (low disk space / ARM64),
                    2 = unsupported (32-bit Windows).

check-dlib          Exit 0 when THIS interpreter can import dlib, i.e. the
                    face recognition engine does not have to be compiled
                    again. Used to decide whether a C++ toolkit is needed.

detect-cmake        Look for a usable CMake: on PATH, in the standard
                    install locations and in the CMake copy that ships with
                    Visual Studio. Exit 0 when one works. With --env-file
                    it also writes IAMS_CMAKE_OK and IAMS_CMAKE_BIN for
                    setup.bat.

detect-msvc         Check the components dlib actually needs - the MSVC
                    cl.exe compiler, link.exe and the Windows SDK headers
                    and libraries - instead of just looking for a Build
                    Tools product. Exit 0 when all of them are present.
                    With --env-file it also writes IAMS_MSVC_OK,
                    IAMS_VS_HAS_CPP, IAMS_VS_INSTANCE and IAMS_VS_INSTALLER.

verify-install      Import every runtime-critical package and exit 0 only
                    if all of them import successfully. Prints one line per
                    package so failures are immediately diagnosable.
                    Pass --quiet for a silent exit-code-only check, and
                    --with-dlib to also check the optional dlib recogniser.

Exit codes: 0 = OK, 1 = failed / warning, 2 = unsupported or usage error.
"""

import glob
import os
import platform
import shutil
import subprocess
import sys
import warnings

SUPPORTED_VERSIONS = ((3, 10), (3, 11), (3, 12))
RECOMMENDED_VERSION = (3, 10)

# The C++ workload and MSVC component IDs come from the official
# "Visual Studio Build Tools workload and component IDs" page.
VC_TOOLS_COMPONENT = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"

# Free space below this is reported as a warning; the "Desktop development
# with C++" workload needs several GB.
MIN_FREE_DISK_GB = 15.0

# (module to import, distribution name, human-readable label)
CRITICAL_PACKAGES = [
    ("streamlit", "streamlit", "streamlit"),
    ("cv2", "opencv-python", "opencv-python"),
    ("mediapipe", "mediapipe", "mediapipe"),
    ("numpy", "numpy", "numpy"),
    ("pandas", "pandas", "pandas"),
    ("PIL", "Pillow", "Pillow"),
    ("fpdf", "fpdf2", "fpdf2"),
    ("dotenv", "python-dotenv", "python-dotenv"),
    ("requests", "requests", "requests"),
]

# Optional dlib recogniser (requirements-dlib.txt, setup.bat --with-dlib);
# checked only by `verify-install --with-dlib`.
DLIB_PACKAGES = [
    ("face_recognition", "face-recognition", "face-recognition (dlib)"),
    # face_recognition_models does `from pkg_resources import resource_filename`
    # at import time, so the venv must contain a setuptools that still ships
    # pkg_resources (setuptools < 81).
    (None, "setuptools", "setuptools (provides pkg_resources)"),
]

# A project folder path longer than this can push files deep inside .venv
# past Windows' 260-character path limit (unless long paths are enabled).
LONG_PATH_WARNING = 100


def _installed_version(distribution_name):
    """Best-effort version lookup via importlib.metadata (3.8+)."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python 3.7 and older
        return "unknown"
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return "unknown"
    except Exception:
        return "unknown"


def _run(cmd, timeout=90):
    """Run a helper executable and return (returncode, combined output)."""
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except Exception as exc:
        return 1, "could not run %s: %s" % (cmd[0], exc)
    return proc.returncode, proc.stdout.decode("utf-8", "replace")


def _write_env_file(path, values):
    """Write a small batch file that setup.bat can `call`.

    This avoids parsing human-readable output inside cmd.exe. Paths are
    written with CRLF line endings, which batch files prefer.
    """
    try:
        directory = os.path.dirname(os.path.abspath(path))
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
    except Exception:
        pass

    lines = [
        "@echo off",
        "REM Generated by scripts/setup_helper.py - safe to delete.",
    ]
    for name, value in values:
        if value:
            lines.append('set "%s=%s"' % (name, value))
        else:
            lines.append('set "%s="' % name)
    try:
        with open(path, "w", encoding="ascii", errors="backslashreplace",
                  newline="\r\n") as handle:
            handle.write("\n".join(lines) + "\n")
    except Exception as exc:
        print("WARNING: could not write %s: %s" % (path, exc))
        return False
    return True


# ----------------------------------------------------------------------
# Python version
# ----------------------------------------------------------------------
def check_python(argv):
    quiet = "--quiet" in argv
    current = (sys.version_info.major, sys.version_info.minor)

    if current in SUPPORTED_VERSIONS:
        if not quiet:
            print(
                "OK: Python %s at %s - supported"
                % (sys.version.split()[0], sys.executable)
            )
            if current != RECOMMENDED_VERSION:
                print(
                    "     note: 3.10 is the version this project is verified on;\n"
                    "     %d.%d satisfies the dependency constraints but was not\n"
                    "     tested on this machine."
                    % current
                )
        return 0

    print("ERROR: Python %d.%d is not supported by this project."
          % current)
    print("       Supported: 3.10 (verified), 3.11, 3.12 (constraints checked).")
    print("       Python 3.13 and newer are not supported: face_recognition")
    print("       installation failed on 3.14 in this project - see")
    print("       PHASE_2A_INSTALLATION_NOTES.txt.")
    print("       Recommended: Python 3.10 from")
    print("       https://www.python.org/downloads/windows/")
    print("       (tick 'Add Python to PATH' in the installer).")
    return 1


def check_venv(argv):
    """Supported Python and no damaged package metadata in this environment."""
    if check_python(["--quiet"]) != 0:
        print("The .venv was created with Python %d.%d, which is not supported."
              % sys.version_info[:2])
        return 1

    import sysconfig
    damaged = []
    for key in ("purelib", "platlib"):
        folder = sysconfig.get_paths().get(key)
        if not folder or not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if (name.endswith(".dist-info") and os.path.isdir(path)
                    and not os.path.isfile(os.path.join(path, "METADATA"))
                    and name not in damaged):
                damaged.append(name)
    if damaged:
        print("The .venv has damaged package installs (METADATA file missing),")
        print("usually left by an interrupted install or removed by antivirus:")
        for name in damaged[:10]:
            print("    " + name)
        if len(damaged) > 10:
            print("    ... and %d more" % (len(damaged) - 10))
        return 1
    return 0


# ----------------------------------------------------------------------
# Windows / architecture / disk space
# ----------------------------------------------------------------------
def check_windows(argv):
    quiet = "--quiet" in argv

    arch = os.environ.get("PROCESSOR_ARCHITECTURE") or platform.machine()
    native = os.environ.get("PROCESSOR_ARCHITEW6432")
    effective = native or arch

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    drive = os.path.splitdrive(root)[0] or os.path.splitdrive(os.getcwd())[0]

    free_gb = None
    try:
        usage = shutil.disk_usage(drive + os.sep if drive else root)
        free_gb = usage.free / (1024.0 ** 3)
    except Exception:
        pass

    if not quiet:
        print("Windows       : %s" % platform.platform())
        print("Architecture  : %s" % effective)
        if free_gb is not None:
            print("Free disk (%s): %.1f GB" % (drive or "?", free_gb))

    if effective.lower() in ("x86", "i386", "i686"):
        print()
        print("ERROR: 32-bit Windows is not supported by this project.")
        print("       Python 64-bit and the Microsoft C++ build tools are")
        print("       64-bit only. 64-bit Windows 10 or 11 is required.")
        return 2

    warning = False
    if effective.lower() == "arm64":
        warning = True
        print()
        print("NOTE: this is an ARM64 Windows PC. x64 Python and the C++")
        print("      build tools run under emulation, which is slower but")
        print("      usually works for compiling dlib.")
    if free_gb is not None and free_gb < MIN_FREE_DISK_GB:
        warning = True
        print()
        print("WARNING: only %.1f GB free on %s." % (free_gb, drive or "this drive"))
        print("         The Microsoft C++ build tools need roughly 3 to 7 GB")
        print("         when they have to be installed.")
    if len(root) > LONG_PATH_WARNING and not _long_paths_enabled():
        warning = True
        print()
        print("WARNING: the project folder path is %d characters long:" % len(root))
        print("           %s" % root)
        print("         Files deep inside .venv can then exceed Windows' 260-")
        print("         character path limit, and package installs fail with")
        print("         'No such file or directory' or 'legacy-install-failure'.")
        print("         Move the project to a short folder such as C:\\IAMS - when")
        print("         unzipping, do not extract into a folder of the same name.")
    if not warning and not quiet:
        print("Result        : this PC meets the basic requirements.")
    return 1 if warning else 0


def _long_paths_enabled():
    """True when Windows long paths (LongPathsEnabled) are switched on."""
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"SYSTEM\CurrentControlSet\Control\FileSystem")
        value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
        return value == 1
    except Exception:
        return False


# ----------------------------------------------------------------------
# dlib availability in the target environment
# ----------------------------------------------------------------------
def check_dlib(argv):
    quiet = "--quiet" in argv
    try:
        import dlib
    except Exception as exc:
        if not quiet:
            print("dlib is NOT available: %s: %s" % (type(exc).__name__, exc))
        return 1

    version = getattr(dlib, "__version__", None) or _installed_version("dlib")
    if not quiet:
        print("dlib %s is already importable - no C++ build is needed." % version)
    return 0


# ----------------------------------------------------------------------
# CMake detection
# ----------------------------------------------------------------------
def _cmake_candidates():
    candidates = []

    found = shutil.which("cmake")
    if found:
        candidates.append(found)

    for variable in ("ProgramFiles", "ProgramFiles(x86)", "ProgramData",
                     "LOCALAPPDATA"):
        base = os.environ.get(variable)
        if not base:
            continue
        candidates.append(os.path.join(base, "CMake", "bin", "cmake.exe"))
        candidates.append(os.path.join(base, "Programs", "CMake", "bin",
                                       "cmake.exe"))

    # CMake that is installed as part of the Visual Studio C++ workload.
    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(variable)
        if not base:
            continue
        pattern = os.path.join(
            base, "Microsoft Visual Studio", "*", "*", "Common7", "IDE",
            "CommonExtensions", "Microsoft", "CMake", "CMake", "bin",
            "cmake.exe",
        )
        candidates.extend(glob.glob(pattern))

    seen = set()
    unique = []
    for path in candidates:
        key = path.lower()
        if key in seen:
            continue
        seen.add(key)
        if os.path.isfile(path):
            unique.append(path)
    return unique


def find_cmake():
    """Return (path, version) of a working CMake, or (None, None)."""
    for exe in _cmake_candidates():
        code, output = _run([exe, "--version"], timeout=60)
        if code != 0:
            continue
        first_line = output.strip().splitlines()[0] if output.strip() else ""
        if "cmake version" not in first_line.lower():
            continue
        version = first_line.lower().split("cmake version", 1)[1].strip()
        return exe, version.split()[0] if version else "unknown"
    return None, None


def detect_cmake(argv):
    quiet = "--quiet" in argv
    env_file = _arg_value(argv, "--env-file")

    exe, version = find_cmake()

    if exe:
        if not quiet:
            print("CMake         : FOUND")
            print("Version       : %s" % version)
            print("Executable    : %s" % exe)
            print("Result        : dlib can be generated and built with CMake.")
        if env_file:
            _write_env_file(env_file, [
                ("IAMS_CMAKE_OK", "1"),
                ("IAMS_CMAKE_BIN", os.path.dirname(exe)),
            ])
        return 0

    if not quiet:
        print("CMake         : NOT FOUND")
        print("Checked       : PATH, the standard CMake install folders and")
        print("                the CMake copy shipped with Visual Studio.")
        print("Result        : CMake is missing. pip needs it to generate the")
        print("                Visual Studio build files for dlib.")
    if env_file:
        _write_env_file(env_file, [
            ("IAMS_CMAKE_OK", "0"),
            ("IAMS_CMAKE_BIN", ""),
        ])
    return 1


# ----------------------------------------------------------------------
# Microsoft C++ build tools detection
# ----------------------------------------------------------------------
def _vswhere_path():
    for variable in ("ProgramFiles(x86)", "ProgramFiles"):
        base = os.environ.get(variable)
        if not base:
            continue
        candidate = os.path.join(base, "Microsoft Visual Studio", "Installer",
                                 "vswhere.exe")
        if os.path.isfile(candidate):
            return candidate
    return None


def _vswhere_instances(requires=None):
    """Installed Visual Studio instances (optionally filtered by component).

    Returns None when vswhere is unavailable so callers can fall back to a
    filesystem scan.
    """
    exe = _vswhere_path()
    if not exe:
        return None
    cmd = [exe, "-products", "*"]
    if requires:
        cmd += ["-requires", requires]
    cmd += ["-format", "value", "-property", "installationPath"]
    code, output = _run(cmd, timeout=90)
    if code != 0 and not output.strip():
        return []
    paths = [line.strip() for line in output.splitlines() if line.strip()]
    return [p for p in paths if os.path.isdir(p)]


def _fallback_instances():
    """Scan the standard folders when vswhere is not available."""
    results = []
    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(variable)
        if not base:
            continue
        pattern = os.path.join(base, "Microsoft Visual Studio", "*", "*")
        for path in glob.glob(pattern):
            if os.path.isdir(path):
                results.append(path)
    return results


def _find_msvc_binaries(instances):
    """Return (cl.exe, link.exe) from the newest MSVC toolset found."""
    cl = link = None
    for instance in instances:
        for toolset in sorted(
            glob.glob(os.path.join(instance, "VC", "Tools", "MSVC", "*")),
            reverse=True,
        ):
            for host in ("Hostx64", "Hostx86"):
                for arch in ("x64", "x86"):
                    candidate_cl = os.path.join(toolset, "bin", host, arch,
                                                "cl.exe")
                    candidate_link = os.path.join(toolset, "bin", host, arch,
                                                  "link.exe")
                    if os.path.isfile(candidate_cl) and os.path.isfile(candidate_link):
                        if cl is None:
                            cl, link = candidate_cl, candidate_link
    return cl, link


def _find_windows_sdk():
    """Return a dict describing a Windows SDK usable by dlib, or None."""
    for variable in ("ProgramFiles(x86)", "ProgramFiles"):
        base = os.environ.get(variable)
        if not base:
            continue
        kits = os.path.join(base, "Windows Kits", "10")
        lib_root = os.path.join(kits, "Lib")
        if not os.path.isdir(lib_root):
            continue
        for version in sorted(os.listdir(lib_root), reverse=True):
            kernel32 = os.path.join(lib_root, version, "um", "x64",
                                    "kernel32.lib")
            ucrt = os.path.join(lib_root, version, "ucrt", "x64", "ucrt.lib")
            stdlib = os.path.join(kits, "Include", version, "ucrt",
                                  "stdlib.h")
            if os.path.isfile(kernel32) and os.path.isfile(ucrt) \
                    and os.path.isfile(stdlib):
                return {
                    "version": version,
                    "kits": kits,
                    "kernel32": kernel32,
                    "ucrt": ucrt,
                }
    return None


def detect_msvc(argv):
    quiet = "--quiet" in argv
    env_file = _arg_value(argv, "--env-file")

    with_cpp = _vswhere_instances(VC_TOOLS_COMPONENT)
    all_instances = _vswhere_instances()
    if all_instances is None:
        all_instances = _fallback_instances()
        with_cpp = None  # cannot filter by component without vswhere
        search = all_instances
    else:
        search = with_cpp if with_cpp else all_instances

    cl, link = _find_msvc_binaries(search if search else all_instances)
    sdk = _find_windows_sdk()

    # The instance setup.bat may extend with the C++ workload.
    instance_for_modify = ""
    if all_instances:
        instance_for_modify = all_instances[0]
    installer = ""
    vswhere = _vswhere_path()
    if vswhere:
        candidate = os.path.join(os.path.dirname(vswhere), "setup.exe")
        if os.path.isfile(candidate):
            installer = candidate

    ok = bool(cl and link and sdk)

    if not quiet:
        print("Microsoft C++ build tools")
        print()
        if with_cpp:
            print("Visual Studio instances with the C++ workload:")
            for path in with_cpp:
                print("  - %s" % path)
        elif all_instances:
            print("Visual Studio installations found, but the 'Desktop")
            print("development with C++' workload is not installed:")
            for path in all_instances:
                print("  - %s" % path)
        else:
            print("No Visual Studio installation was found.")
        print()
        print("  C++ compiler  : %s" % ("FOUND  %s" % cl if cl else "MISSING"))
        print("  C++ linker    : %s" % ("FOUND  %s" % link if link else "MISSING"))
        if sdk:
            print("  Windows SDK   : FOUND  %s" % sdk["version"])
            print("                  %s" % sdk["kernel32"])
            print("                  %s" % sdk["ucrt"])
        else:
            print("  Windows SDK   : MISSING  (no Windows Kits 10 with")
            print("                  um\\x64\\kernel32.lib, ucrt\\x64\\ucrt.lib")
            print("                  and the ucrt headers)")
        print()
        if ok:
            print("Result        : everything dlib needs to compile is present.")
        else:
            print("Result        : the Microsoft C++ build tools are NOT")
            print("                complete. The missing pieces above are")
            print("                required to compile dlib from source.")
            print("                Installing the Build Tools product alone is")
            print("                not enough - the workload 'Desktop")
            print("                development with C++' plus a Windows SDK")
            print("                must be selected.")

    if env_file:
        _write_env_file(env_file, [
            ("IAMS_MSVC_OK", "1" if ok else "0"),
            ("IAMS_VS_HAS_CPP", "1" if with_cpp else "0"),
            ("IAMS_VS_INSTANCE", instance_for_modify),
            ("IAMS_VS_INSTALLER", installer),
        ])

    return 0 if ok else 1


def _arg_value(argv, name):
    """Return the value of ``--name value`` or ``--name=value``."""
    for index, item in enumerate(argv):
        if item == name and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith(name + "="):
            return item.split("=", 1)[1]
    return None


# ----------------------------------------------------------------------
# Project dependency verification
# ----------------------------------------------------------------------
def _package_version(module, distribution_name):
    module_version = getattr(module, "__version__", None)
    if module_version:
        return str(module_version)
    return _installed_version(distribution_name)


def verify_install(argv):
    quiet = "--quiet" in argv
    if not quiet:
        print("Interpreter : Python %s" % sys.version.split()[0])
        print("Executable  : %s" % sys.executable)
        in_venv = sys.prefix != sys.base_prefix
        print("Environment : %s"
              % ("virtual environment" if in_venv
                 else "system interpreter (not a venv)"))
        print()

    # pkg_resources emits a DeprecationWarning on modern setuptools; that
    # warning is expected and should not pollute the setup output.
    warnings.filterwarnings("ignore")

    packages = CRITICAL_PACKAGES + (DLIB_PACKAGES if "--with-dlib" in argv else [])
    failures = []
    for module_name, distribution_name, label in packages:
        try:
            if module_name is None:
                module = __import__("pkg_resources")
            else:
                module = __import__(module_name)
            version = _package_version(module, distribution_name)
            if not quiet:
                print("  OK    %-34s %s" % (label, version))
        except Exception as exc:
            failures.append((label, distribution_name, exc))
            if not quiet:
                print("  FAIL  %-34s %s: %s"
                      % (label, type(exc).__name__, exc))

    if failures:
        if not quiet:
            print()
            print("ERROR: %d critical package(s) failed to import:" % len(failures))
            for label, distribution_name, exc in failures:
                print("  - %s (%s): %s" % (label, distribution_name, exc))
            print()
            print("Fix hints:")
            print("  - Re-run setup.bat to reinstall the dependencies.")
            print("  - 'No module named pkg_resources': the venv contains")
            print("    setuptools >= 81 which removed pkg_resources; setup.bat")
            print("    installs 'setuptools<81' from requirements.txt to fix this.")
            print("  - dlib / face-recognition (only with --with-dlib): compiling")
            print("    dlib needs the Microsoft C++ build tools and CMake (see README.md).")
        return 1

    if not quiet:
        print()
        print("All %d critical packages imported successfully."
              % len(packages))
    return 0


COMMANDS = {
    "check-venv": check_venv,
    "check-python": check_python,
    "check-windows": check_windows,
    "check-dlib": check_dlib,
    "detect-cmake": detect_cmake,
    "detect-msvc": detect_msvc,
    "verify-install": verify_install,
}


def main(argv):
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        print("usage: setup_helper.py "
              "{check-python|check-venv|check-windows|check-dlib|detect-cmake|"
              "detect-msvc|verify-install} [--quiet] [--env-file PATH]")
        return 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
