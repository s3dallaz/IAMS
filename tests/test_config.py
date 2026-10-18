"""
Test configuration module

Phase 1A: Test that configuration can be imported and validated
"""

import pytest
from pathlib import Path


def test_config_import():
    """Test that config module can be imported successfully."""
    import config
    assert config is not None


def test_config_constants():
    """Test that required configuration constants exist."""
    import config
    
    # Test required constants
    assert hasattr(config, "APP_NAME")
    assert hasattr(config, "APP_VERSION")
    assert hasattr(config, "PROJECT_ROOT")
    assert hasattr(config, "DATA_DIR")
    assert hasattr(config, "DATABASE_PATH")
    
    # Test types
    assert isinstance(config.APP_NAME, str)
    assert isinstance(config.APP_VERSION, str)
    assert isinstance(config.PROJECT_ROOT, Path)
    assert isinstance(config.DATA_DIR, Path)
    assert isinstance(config.DATABASE_PATH, Path)


def test_config_paths():
    """Test that configuration paths are properly constructed."""
    import config
    
    # Test path relationships
    assert config.DATA_DIR == config.PROJECT_ROOT / "data"
    assert config.DATABASE_DIR == config.DATA_DIR / "database"
    assert config.FACE_ENCODINGS_DIR == config.DATA_DIR / "face_encodings"
    assert config.LOGS_DIR == config.DATA_DIR / "logs"


def test_ensure_directories():
    """Test that ensure_directories creates required directories."""
    import config
    
    config.ensure_directories()
    
    # Verify directories exist
    assert config.DATA_DIR.exists()
    assert config.DATABASE_DIR.exists()
    assert config.FACE_ENCODINGS_DIR.exists()
    assert config.ENROLLMENT_IMAGES_DIR.exists()
    assert config.UNKNOWN_FACES_DIR.exists()
    assert config.LOGS_DIR.exists()


def test_validate_config():
    """Test configuration validation function."""
    import config
    
    result = config.validate_config()
    assert result is True


def test_src_imports():
    """Test that src modules can be imported."""
    import src
    import src.database
    import src.services
    import src.computer_vision
    import src.ai
    import src.ui
    import src.utils
    
    # Verify package version
    assert hasattr(src, "__version__")
    assert isinstance(src.__version__, str)


def test_logger_import():
    """Test that logger utility can be imported and used."""
    from src.utils.logger import get_logger, setup_logger
    
    # Create a test logger
    logger = get_logger("test")
    assert logger is not None
    assert logger.name == "test"
    
    # Test that logger can log without errors
    logger.info("Test log message")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def _msmf_flag_after_importing_config(preset=None):
    """Import config in a fresh interpreter and report the MSMF flag value."""
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items()
           if k != "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"}
    if preset is not None:
        env["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = preset
    code = ("import config, os; "
            "print(os.environ.get('OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'))")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parent.parent,
        env=env, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def test_config_disables_msmf_hw_transforms_for_fast_camera_start():
    """Importing config sets the flag that makes the camera open in ~1 s."""
    assert _msmf_flag_after_importing_config() == "0"


def test_config_keeps_user_msmf_hw_transforms_setting():
    """A value set in the environment (or .env) is not overwritten."""
    assert _msmf_flag_after_importing_config(preset="1") == "1"


def test_face_recognition_threshold_can_be_set_from_environment():
    """FACE_RECOGNITION_THRESHOLD in .env/environment overrides the 0.6 default."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "FACE_RECOGNITION_THRESHOLD": "0.5"}
    result = subprocess.run(
        [sys.executable, "-c", "import config; print(config.FACE_RECOGNITION_THRESHOLD)"],
        cwd=Path(__file__).resolve().parent.parent,
        env=env, capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "0.5"


def test_face_recognition_threshold_default_is_unchanged():
    """Without an override the threshold stays at the original 0.6."""
    import os
    import config

    if "FACE_RECOGNITION_THRESHOLD" in os.environ:
        pytest.skip("FACE_RECOGNITION_THRESHOLD is set in this environment")
    assert config.FACE_RECOGNITION_THRESHOLD == 0.6
