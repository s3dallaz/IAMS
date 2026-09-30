"""
Computer Vision dependency verification utility.

Verifies that all required CV dependencies are installed and accessible.
This is used for diagnostics and testing.
"""

import sys
from typing import Dict, Any, Optional


def verify_cv_dependencies() -> Dict[str, Any]:
    """
    Verify that all required CV dependencies can be imported.
    
    Returns:
        Dictionary containing verification results:
        {
            'python_version': str,
            'dependencies': {
                'opencv': {'installed': bool, 'version': str or None, 'error': str or None},
                'face_recognition': {...},
                'mediapipe': {...},
                'numpy': {...},
                'pillow': {...}
            },
            'all_available': bool
        }
    """
    results = {
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'dependencies': {},
        'all_available': True
    }
    
    # Check OpenCV
    try:
        import cv2
        results['dependencies']['opencv'] = {
            'installed': True,
            'version': cv2.__version__,
            'error': None
        }
    except ImportError as e:
        results['dependencies']['opencv'] = {
            'installed': False,
            'version': None,
            'error': str(e)
        }
        results['all_available'] = False
    
    # Check face_recognition
    try:
        import face_recognition
        results['dependencies']['face_recognition'] = {
            'installed': True,
            'version': getattr(face_recognition, '__version__', 'unknown'),
            'error': None
        }
    except (ImportError, SystemExit) as e:
        # SystemExit can occur with face_recognition_models compatibility issues
        error_msg = str(e) if not isinstance(e, SystemExit) else 'face_recognition_models compatibility issue'
        results['dependencies']['face_recognition'] = {
            'installed': False,
            'version': None,
            'error': error_msg
        }
        results['all_available'] = False
    
    # Check MediaPipe
    try:
        import mediapipe
        results['dependencies']['mediapipe'] = {
            'installed': True,
            'version': mediapipe.__version__,
            'error': None
        }
    except ImportError as e:
        results['dependencies']['mediapipe'] = {
            'installed': False,
            'version': None,
            'error': str(e)
        }
        results['all_available'] = False
    
    # Check NumPy
    try:
        import numpy
        results['dependencies']['numpy'] = {
            'installed': True,
            'version': numpy.__version__,
            'error': None
        }
    except ImportError as e:
        results['dependencies']['numpy'] = {
            'installed': False,
            'version': None,
            'error': str(e)
        }
        results['all_available'] = False
    
    # Check Pillow
    try:
        import PIL
        results['dependencies']['pillow'] = {
            'installed': True,
            'version': PIL.__version__,
            'error': None
        }
    except ImportError as e:
        results['dependencies']['pillow'] = {
            'installed': False,
            'version': None,
            'error': str(e)
        }
        results['all_available'] = False
    
    return results


def print_verification_report(results: Optional[Dict[str, Any]] = None) -> None:
    """
    Print a formatted verification report.
    
    Args:
        results: Verification results dictionary. If None, will run verification.
    """
    if results is None:
        results = verify_cv_dependencies()
    
    print(f"Python version: {results['python_version']}")
    print("\nComputer Vision Dependencies:")
    print("-" * 60)
    
    for dep_name, dep_info in results['dependencies'].items():
        status = "[OK] INSTALLED" if dep_info['installed'] else "[X] MISSING"
        version = f"v{dep_info['version']}" if dep_info['version'] else "N/A"
        
        print(f"{dep_name:20} {status:15} {version}")
        
        if dep_info['error']:
            print(f"  Error: {dep_info['error']}")
    
    print("-" * 60)
    
    if results['all_available']:
        print("\n[OK] All CV dependencies are available")
    else:
        print("\n[ERROR] Some CV dependencies are missing")
    
    return results['all_available']


if __name__ == "__main__":
    success = print_verification_report()
    sys.exit(0 if success else 1)
