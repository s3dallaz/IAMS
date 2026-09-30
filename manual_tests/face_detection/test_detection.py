"""
Minimal Face Detection Testing Script

Tests the FaceDetector by loading images and displaying detected faces with bounding boxes.
Press any key to navigate through images, ESC to exit.
"""

import sys
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import cv2
from src.computer_vision.face_detector import FaceDetector


def draw_bounding_boxes(image, detected_faces):
    """
    Draw bounding boxes on the image for all detected faces.
    
    Args:
        image: Original image (will be copied)
        detected_faces: List of DetectedFace objects
    
    Returns:
        Image with bounding boxes drawn
    """
    result = image.copy()
    
    for idx, face in enumerate(detected_faces, 1):
        x, y, w, h = face.bbox
        
        # Draw rectangle
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw label with face number and confidence
        label = f"Face {idx}: {face.confidence:.3f}"
        
        # Draw label background
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(result, (x, y - label_h - 10), (x + label_w, y), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(result, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return result


def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from a directory."""
    if not directory.exists():
        return []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    return sorted([
        f for f in directory.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ])


def test_images_in_folder(detector: FaceDetector, folder_path: Path, folder_name: str):
    """
    Test all images in a folder and display results.
    
    Args:
        detector: FaceDetector instance
        folder_path: Path to the folder containing images
        folder_name: Name of the folder (for display)
    """
    print(f"\n{'='*70}")
    print(f"Testing: {folder_name}/")
    print(f"{'='*70}")
    
    image_files = get_image_files(folder_path)
    
    if not image_files:
        print(f"[INFO] No images found in {folder_name}/")
        return
    
    print(f"[INFO] Found {len(image_files)} image(s)\n")
    
    for image_file in image_files:
        print(f"\n{'─'*70}")
        print(f"Image: {image_file.name}")
        print(f"{'─'*70}")
        
        # Load image
        try:
            image = cv2.imread(str(image_file))
            if image is None:
                print("[FAILED] Could not load image (invalid format or corrupted)")
                continue
        except Exception as e:
            print(f"[ERROR] Failed to load image: {e}")
            continue
        
        # Detect faces
        try:
            detected_faces = detector.detect_faces(image)
        except Exception as e:
            print(f"[ERROR] Face detection failed: {e}")
            continue
        
        # Print results
        print(f"[OK] Detected {len(detected_faces)} face(s)")
        
        if detected_faces:
            for idx, face in enumerate(detected_faces, 1):
                x, y, w, h = face.bbox
                print(f"     Face {idx}: bbox=({x}, {y}, {w}, {h}), confidence={face.confidence:.3f}")
            
            # Draw bounding boxes
            result_image = draw_bounding_boxes(image, detected_faces)
        else:
            result_image = image.copy()
            # Add text overlay for no faces
            cv2.putText(result_image, "No faces detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display image
        window_name = f"Face Detection - {image_file.name} - Press any key for next, ESC to exit"
        cv2.imshow(window_name, result_image)
        
        # Wait for key press
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # ESC key to exit
        if key == 27:
            print("\n[INFO] Exiting...")
            return


def main():
    """Main entry point for face detection testing."""
    print("\n" + "="*70)
    print("FACE DETECTION TEST")
    print("="*70)
    print("\nThis script tests the FaceDetector on images in test folders.")
    print("Press any key to navigate through images, ESC to exit.\n")
    
    # Define test folders
    test_dir = Path(__file__).parent
    folders = [
        (test_dir / "single_face", "single_face"),
        (test_dir / "multiple_faces", "multiple_faces"),
        (test_dir / "no_face", "no_face")
    ]
    
    # Initialize detector
    print("[INFO] Initializing FaceDetector...")
    try:
        detector = FaceDetector()
        print("[OK] FaceDetector initialized\n")
    except Exception as e:
        print(f"[FAILED] Could not initialize FaceDetector: {e}")
        return
    
    try:
        # Test each folder
        for folder_path, folder_name in folders:
            test_images_in_folder(detector, folder_path, folder_name)
        
        print("\n" + "="*70)
        print("TESTING COMPLETE")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Stopped by user")
    
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        detector.close()
        cv2.destroyAllWindows()
        print("\n[OK] Resources released")


if __name__ == "__main__":
    main()
