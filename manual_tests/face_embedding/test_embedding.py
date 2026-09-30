"""
Minimal Face Embedding Testing Script

Tests the FaceEmbedder by generating 128-dimensional embeddings from face images.
Uses FaceDetector to extract faces, then FaceEmbedder to generate embeddings.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from src.computer_vision.face_detector import FaceDetector
from src.computer_vision.face_embedder import FaceEmbedder, FaceEmbeddingError


def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from a directory."""
    if not directory.exists():
        return []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    return sorted([
        f for f in directory.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ])


def process_image(
    image_path: Path, 
    detector: FaceDetector, 
    embedder: FaceEmbedder,
    save_debug_crops: bool = False
) -> Tuple[bool, Optional[np.ndarray], str]:
    """
    Process a single image and generate embedding.
    
    Args:
        image_path: Path to the image file
        detector: FaceDetector instance
        embedder: FaceEmbedder instance
        save_debug_crops: Whether to save face crops for debugging
    
    Returns:
        Tuple of (success, embedding, status_message)
    """
    # Load image
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return False, None, "Failed to load image (invalid format)"
    except Exception as e:
        return False, None, f"Error loading image: {e}"
    
    # Detect faces with MediaPipe
    try:
        detected_faces = detector.detect_faces(image)
    except Exception as e:
        return False, None, f"Face detection error: {e}"
    
    # Check face count
    if len(detected_faces) == 0:
        return False, None, "No face detected"
    
    if len(detected_faces) > 1:
        return False, None, f"Multiple faces detected ({len(detected_faces)})"
    
    # Get face bounding box from MediaPipe
    face = detected_faces[0]
    x, y, w, h = face.bbox
    
    # Optionally save debug crop
    if save_debug_crops:
        face_crop = image[y:y+h, x:x+w].copy()
        debug_dir = Path(__file__).parent / "debug_crops"
        debug_dir.mkdir(exist_ok=True)
        crop_path = debug_dir / f"{image_path.stem}_crop.jpg"
        cv2.imwrite(str(crop_path), face_crop)
        print(f"     Debug crop saved: {crop_path}")
    
    # Generate embedding using full image with known face location
    # This is more reliable than cropping, as dlib sometimes fails to detect
    # faces in small or tightly cropped images
    try:
        # Convert BGR to RGB (face_recognition expects RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # face_recognition.face_encodings expects face locations as (top, right, bottom, left)
        face_location = (y, x + w, y + h, x)
        
        # Import face_recognition here to use known_face_locations parameter
        import face_recognition
        encodings = face_recognition.face_encodings(rgb_image, known_face_locations=[face_location])
        
        if len(encodings) == 0:
            return False, None, "Could not generate embedding from detected face"
        
        embedding = encodings[0]
        
        # Verify dimensions
        if embedding.shape != (128,):
            return False, None, f"Invalid embedding shape: {embedding.shape}"
        
        return True, embedding, f"Success (detection confidence: {face.confidence:.3f})"
        
    except Exception as e:
        return False, None, f"Embedding error: {e}"


def calculate_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate Euclidean distance between two embeddings."""
    return np.linalg.norm(embedding1 - embedding2)


def main():
    """Main entry point for face embedding testing."""
    print("\n" + "="*70)
    print("FACE EMBEDDING TEST")
    print("="*70)
    print("\nThis script generates 128-dimensional face embeddings from images.")
    print("Images must contain exactly one face.\n")
    
    # Define images directory
    test_dir = Path(__file__).parent
    images_dir = test_dir / "images"
    
    print(f"Images directory: {images_dir}\n")
    
    # Check if directory exists
    if not images_dir.exists():
        print(f"[FAILED] Directory not found: {images_dir}")
        return
    
    # Get image files
    image_files = get_image_files(images_dir)
    
    if not image_files:
        print(f"[INFO] No images found in {images_dir}/")
        print("       Add .jpg, .png, or other image files to test")
        return
    
    print(f"[INFO] Found {len(image_files)} image(s)\n")
    
    # Initialize detector and embedder
    print("[INFO] Initializing FaceDetector and FaceEmbedder...")
    try:
        detector = FaceDetector()
        embedder = FaceEmbedder()
        print("[OK] Components initialized\n")
    except Exception as e:
        print(f"[FAILED] Could not initialize components: {e}")
        return
    
    try:
        # Process all images
        print("="*70)
        print("PROCESSING IMAGES")
        print("="*70)
        
        results = []
        
        for image_file in image_files:
            print(f"\n{image_file.name}")
            print("-" * len(image_file.name))
            
            success, embedding, message = process_image(image_file, detector, embedder)
            
            if success:
                print(f"[OK] {message}")
                print(f"     Embedding shape: {embedding.shape}")
                print(f"     Embedding dtype: {embedding.dtype}")
                results.append((image_file.name, embedding))
            else:
                print(f"[FAILED] {message}")
                results.append((image_file.name, None))
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        successful = sum(1 for _, emb in results if emb is not None)
        print(f"\nTotal images: {len(results)}")
        print(f"Successful embeddings: {successful}")
        print(f"Failed: {len(results) - successful}")
        
        # Compare embeddings if we have at least 2 successful ones
        valid_embeddings = [(name, emb) for name, emb in results if emb is not None]
        
        if len(valid_embeddings) >= 2:
            print("\n" + "="*70)
            print("EMBEDDING COMPARISONS")
            print("="*70)
            print("\nEuclidean distances between embeddings:")
            print("(Lower distance = more similar faces)\n")
            
            for i in range(len(valid_embeddings)):
                for j in range(i + 1, len(valid_embeddings)):
                    name1, emb1 = valid_embeddings[i]
                    name2, emb2 = valid_embeddings[j]
                    distance = calculate_distance(emb1, emb2)
                    print(f"{name1} <-> {name2}")
                    print(f"  Distance: {distance:.4f}\n")
        
        print("="*70)
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
        cv2.destroyAllWindows()  # Ensure all OpenCV windows are closed
        print("\n[OK] Resources released")


if __name__ == "__main__":
    main()
