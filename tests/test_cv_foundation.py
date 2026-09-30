"""
Tests for Phase 2A: Computer Vision Foundation

Verifies that:
- CV package can be imported
- Required CV dependencies are available
- CV configuration is properly set up
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config
from src.utils.cv_verification import verify_cv_dependencies


class TestCVPackage:
    """Test computer_vision package structure."""
    
    def test_cv_package_exists(self):
        """Test that computer_vision package can be imported."""
        try:
            import src.computer_vision
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import computer_vision package: {e}")
    
    def test_cv_package_has_version(self):
        """Test that computer_vision package has version attribute."""
        import src.computer_vision
        assert hasattr(src.computer_vision, '__version__')
        assert isinstance(src.computer_vision.__version__, str)


class TestCVDependencies:
    """Test CV dependencies availability."""
    
    def test_opencv_available(self):
        """Test that OpenCV can be imported."""
        try:
            import cv2
            assert hasattr(cv2, '__version__')
        except ImportError as e:
            pytest.fail(f"OpenCV not available: {e}")
    
    def test_face_recognition_available(self):
        """Test that face_recognition can be imported."""
        try:
            import face_recognition
            assert True
        except (ImportError, SystemExit) as e:
            # SystemExit can occur with face_recognition_models compatibility issues
            # This is a known issue with Python 3.14+ (pkg_resources removed)
            if isinstance(e, SystemExit):
                pytest.skip("face_recognition_models incompatible with Python 3.14+ (pkg_resources issue)")
            else:
                pytest.fail(f"face_recognition not available: {e}")
    
    def test_mediapipe_available(self):
        """Test that MediaPipe can be imported."""
        try:
            import mediapipe
            assert hasattr(mediapipe, '__version__')
        except ImportError as e:
            pytest.fail(f"MediaPipe not available: {e}")
    
    def test_numpy_available(self):
        """Test that NumPy can be imported."""
        try:
            import numpy
            assert hasattr(numpy, '__version__')
        except ImportError as e:
            pytest.fail(f"NumPy not available: {e}")
    
    def test_pillow_available(self):
        """Test that Pillow can be imported."""
        try:
            import PIL
            assert hasattr(PIL, '__version__')
        except ImportError as e:
            pytest.fail(f"Pillow not available: {e}")
    
    def test_cv_verification_utility(self):
        """Test CV verification utility works."""
        results = verify_cv_dependencies()
        
        assert 'python_version' in results
        assert 'dependencies' in results
        assert 'all_available' in results
        
        # Check that all required dependencies are reported
        required_deps = ['opencv', 'face_recognition', 'mediapipe', 'numpy', 'pillow']
        for dep in required_deps:
            assert dep in results['dependencies']
            assert 'installed' in results['dependencies'][dep]
            assert 'version' in results['dependencies'][dep]
            assert 'error' in results['dependencies'][dep]
    
    def test_all_cv_dependencies_available(self):
        """Test that all CV dependencies are available."""
        results = verify_cv_dependencies()
        
        if not results['all_available']:
            missing = [
                dep for dep, info in results['dependencies'].items()
                if not info['installed']
            ]
            # Check if the only issue is face_recognition with Python 3.14+
            if missing == ['face_recognition'] and sys.version_info >= (3, 14):
                pytest.skip(f"face_recognition incompatible with Python {sys.version_info.major}.{sys.version_info.minor}+ (known pkg_resources issue)")
            else:
                pytest.fail(f"Missing CV dependencies: {', '.join(missing)}")


class TestCVConfiguration:
    """Test CV-related configuration."""
    
    def test_face_detection_confidence_configured(self):
        """Test that face detection confidence is configured."""
        assert hasattr(config, 'FACE_DETECTION_CONFIDENCE')
        assert isinstance(config.FACE_DETECTION_CONFIDENCE, float)
        assert 0.0 <= config.FACE_DETECTION_CONFIDENCE <= 1.0
    
    def test_face_recognition_threshold_configured(self):
        """Test that face recognition threshold is configured."""
        assert hasattr(config, 'FACE_RECOGNITION_THRESHOLD')
        assert isinstance(config.FACE_RECOGNITION_THRESHOLD, float)
        assert 0.0 <= config.FACE_RECOGNITION_THRESHOLD <= 1.0
    
    def test_cv_data_directories_configured(self):
        """Test that CV-related data directories are configured."""
        assert hasattr(config, 'FACE_ENCODINGS_DIR')
        assert hasattr(config, 'ENROLLMENT_IMAGES_DIR')
        assert hasattr(config, 'UNKNOWN_FACES_DIR')
        
        assert isinstance(config.FACE_ENCODINGS_DIR, Path)
        assert isinstance(config.ENROLLMENT_IMAGES_DIR, Path)
        assert isinstance(config.UNKNOWN_FACES_DIR, Path)
    
    def test_configuration_values_match_architecture(self):
        """Test that configuration values match approved architecture."""
        # Per approved architecture:
        # - recognition threshold: 0.6
        # - detection confidence: 0.7
        assert config.FACE_RECOGNITION_THRESHOLD == 0.6
        assert config.FACE_DETECTION_CONFIDENCE == 0.7


class TestPhase1Integrity:
    """Test that Phase 1 functionality remains intact."""
    
    def test_config_module_still_works(self):
        """Test that config module is still functional."""
        assert hasattr(config, 'APP_NAME')
        assert hasattr(config, 'DATABASE_PATH')
        assert hasattr(config, 'PROJECT_ROOT')
    
    def test_logger_utility_still_works(self):
        """Test that logger utility is still functional."""
        from src.utils.logger import get_logger
        
        logger = get_logger('test_cv_foundation')
        assert logger is not None
    
    def test_database_package_still_works(self):
        """Test that database package is still importable."""
        try:
            from src.database import connection, schema
            assert True
        except ImportError as e:
            pytest.fail(f"Database package import failed: {e}")
    
    def test_services_package_still_works(self):
        """Test that services package is still importable."""
        try:
            from src.services.student_service import StudentService
            assert True
        except ImportError as e:
            pytest.fail(f"Services package import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
