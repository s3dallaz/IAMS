"""
Minimal Camera Test Script

Tests the CameraManager by opening the default webcam and displaying a live feed.
Press 'q' to exit.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
from src.computer_vision.camera_manager import CameraManager


def main():
    """Test camera with live feed display."""
    print("\n" + "="*60)
    print("CAMERA TEST")
    print("="*60)
    print("\nOpening default webcam...")
    print("Press 'q' to exit\n")
    
    camera = CameraManager(camera_index=0)
    
    try:
        # Open camera
        if not camera.open():
            print("[FAILED] Could not open camera")
            print("         Check that your webcam is connected and not in use")
            return
        
        print("[OK] Camera opened successfully")
        
        # Get frame dimensions
        dimensions = camera.get_frame_dimensions()
        if dimensions:
            width, height = dimensions
            print(f"[OK] Resolution: {width}x{height}")
        
        print("\nDisplaying live feed...")
        print("Press 'q' in the video window to exit\n")
        
        # Display live feed
        while True:
            success, frame = camera.read_frame()
            
            if not success or frame is None:
                print("[FAILED] Could not read frame")
                break
            
            # Display frame
            cv2.imshow('Camera Test - Press q to exit', frame)
            
            # Check for 'q' key press (wait 1ms)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[OK] Exiting...")
                break
    
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Stopped by user")
    
    except Exception as e:
        print(f"\n[ERROR] {e}")
    
    finally:
        # Clean up
        camera.release()
        cv2.destroyAllWindows()
        print("[OK] Camera released and windows closed")
        print("\n" + "="*60)


if __name__ == "__main__":
    main()
