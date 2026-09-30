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
