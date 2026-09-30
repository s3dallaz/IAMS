"""
Manual Face Recognition Testing Script

Tests the complete face recognition pipeline:
1. Load images from known/ and unknown/ directories
2. Generate embeddings using FaceDetector and FaceEmbedder
3. Recognize faces using FaceRecognizer
4. Display predicted identities, distances, and match status

Place test images:
- known/student_001/: Multiple images of student 001
- known/student_002/: Multiple images of student 002
- unknown/: Test images to recognize

The script will build a gallery from known/ and test recognition on unknown/.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import face_recognition
from src.computer_vision.face_detector import FaceDetector
from src.computer_vision.face_embedder import FaceEmbedding
from src.computer_vision.face_recognizer import FaceRecognizer


def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from a directory."""
    if not directory.exists():
        return []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    return sorted([
        f for f in directory.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ])


def generate_embedding_from_image(
    image_path: Path,
    detector: FaceDetector
) -> Optional[FaceEmbedding]:
    """
    Generate embedding from an image file.
    
    Returns None if generation fails.
    """
    # Load image
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"     [FAILED] Could not load image")
            return None
    except Exception as e:
        print(f"     [FAILED] Error loading: {e}")
        return None
    
    # Detect face
    try:
        faces = detector.detect_faces(image)
        if len(faces) == 0:
            print(f"     [FAILED] No face detected")
            return None
        if len(faces) > 1:
            print(f"     [FAILED] Multiple faces detected ({len(faces)})")
            return None
    except Exception as e:
        print(f"     [FAILED] Detection error: {e}")
        return None
    
    # Generate embedding using full image with known location
    try:
        face = faces[0]
        x, y, w, h = face.bbox
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_location = (y, x + w, y + h, x)
        
        encodings = face_recognition.face_encodings(rgb_image, known_face_locations=[face_location])
        
        if len(encodings) == 0:
            print(f"     [FAILED] Could not generate embedding")
            return None
        
        return FaceEmbedding(encodings[0])
        
    except Exception as e:
        print(f"     [FAILED] Embedding error: {e}")
        return None


def load_known_faces(known_dir: Path, detector: FaceDetector) -> Dict[str, List[FaceEmbedding]]:
    """
    Load all known face embeddings from subdirectories.
    
    Returns dictionary mapping identity -> list of embeddings.
    """
    print("\n" + "="*70)
    print("LOADING KNOWN FACES")
    print("="*70)
    
    if not known_dir.exists():
        print(f"[ERROR] Directory not found: {known_dir}")
        return {}
    
    known_embeddings = {}
    student_dirs = [d for d in known_dir.iterdir() if d.is_dir()]
    
    if not student_dirs:
        print(f"[INFO] No student directories found in {known_dir}")
        return {}
    
    for student_dir in sorted(student_dirs):
        identity = student_dir.name
        print(f"\n{identity}/")
        
        image_files = get_image_files(student_dir)
        if not image_files:
            print(f"  [INFO] No images found")
            continue
        
        embeddings = []
        for image_file in image_files:
            print(f"  {image_file.name}...", end=" ")
            embedding = generate_embedding_from_image(image_file, detector)
            if embedding:
                embeddings.append(embedding)
                print("[OK]")
            # Error messages already printed by generate_embedding_from_image
        
        if embeddings:
            known_embeddings[identity] = embeddings
            print(f"  => Loaded {len(embeddings)} embedding(s)")
    
    return known_embeddings


def build_gallery(known_embeddings: Dict[str, List[FaceEmbedding]]) -> List[tuple]:
    """
    Build gallery for FaceRecognizer from known embeddings.
    
    Uses the first embedding from each identity (representative).
    Returns list of (identity, embedding) tuples.
    """
    gallery = []
    for identity, embeddings in known_embeddings.items():
        if embeddings:
            # Use first embedding as representative
            gallery.append((identity, embeddings[0]))
    return gallery


def test_unknown_faces(
    unknown_dir: Path,
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    gallery: List[tuple]
):
    """Test recognition on unknown face images."""
    print("\n" + "="*70)
    print("TESTING UNKNOWN FACES")
    print("="*70)
    
    if not gallery:
        print("\n[ERROR] No known faces loaded. Cannot perform recognition.")
        return
    
    print(f"\nGallery contains {len(gallery)} identities")
    
    if not unknown_dir.exists():
        print(f"[ERROR] Directory not found: {unknown_dir}")
        return
    
    image_files = get_image_files(unknown_dir)
    if not image_files:
        print(f"[INFO] No images found in {unknown_dir}")
        return
    
    print(f"Testing {len(image_files)} image(s)...\n")
    
    for image_file in sorted(image_files):
        print(f"\n{'─'*70}")
        print(f"{image_file.name}")
        print(f"{'─'*70}")
        
        # Generate embedding
        embedding = generate_embedding_from_image(image_file, detector)
        if not embedding:
            continue
        
        print(f"[OK] Embedding generated")
        
        # Recognize
        try:
            result = recognizer.recognize(embedding, gallery)
            
            print(f"\nRecognition Result:")
            print(f"  Identity: {result.identity}")
            print(f"  Distance: {result.distance:.4f}")
            print(f"  Match: {'YES' if result.is_match else 'NO'}")
            
            if result.is_match:
                print(f"  => Recognized as {result.identity}")
            else:
                print(f"  => Unknown (closest: {result.distance:.4f}, threshold: {recognizer.threshold})")
        
        except Exception as e:
            print(f"[ERROR] Recognition failed: {e}")


def main():
    """Main entry point for face recognition testing."""
    print("\n" + "="*70)
    print("FACE RECOGNITION TEST")
    print("="*70)
    print("\nThis script tests the complete face recognition pipeline.")
    print("It loads known faces, builds a gallery, and tests recognition.")
    
    # Define directories
    test_dir = Path(__file__).parent
    known_dir = test_dir / "known"
    unknown_dir = test_dir / "unknown"
    
    print(f"\nKnown faces: {known_dir}")
    print(f"Unknown faces: {unknown_dir}")
    
    # Initialize components
    print("\n[INFO] Initializing components...")
    try:
        detector = FaceDetector()
        recognizer = FaceRecognizer()  # Uses default threshold 0.6
        print("[OK] Components initialized")
    except Exception as e:
        print(f"[ERROR] Initialization failed: {e}")
        return
    
    try:
        # Load known faces
        known_embeddings = load_known_faces(known_dir, detector)
        
        if not known_embeddings:
            print("\n[ERROR] No known faces loaded. Add images to known/ subdirectories.")
            return
        
        print(f"\n[OK] Loaded {len(known_embeddings)} identities")
        
        # Build gallery
        gallery = build_gallery(known_embeddings)
        
        # Test unknown faces
        test_unknown_faces(unknown_dir, detector, recognizer, gallery)
        
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
        detector.close()
        cv2.destroyAllWindows()
        print("\n[OK] Resources released")


if __name__ == "__main__":
    main()
