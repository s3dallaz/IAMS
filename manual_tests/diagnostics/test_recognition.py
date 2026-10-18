"""
Diagnostic script: Test recognition with real images and measure performance.
Does not modify any files or database.
"""
from pathlib import Path
import sys
import time
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.services.enrollment_service import EnrollmentService
from src.computer_vision.face_recognizer import FaceRecognizer
from src.computer_vision.face_embedder import FaceEmbedder, FaceEmbedding
from src.computer_vision.face_detector import FaceDetector
import cv2

def pick_test_student(requested_id=None):
    """
    Choose the enrolled student whose enrollment photos are used as the
    "known person" query: the --student-id given on the command line, or
    the first student folder that contains photos.

    Returns (student_id, folder) or (None, None) when nothing is enrolled.
    """
    images_root = project_root / "data" / "enrollment_images"
    if requested_id:
        return requested_id, images_root / requested_id
    if images_root.exists():
        for folder in sorted(p for p in images_root.iterdir() if p.is_dir()):
            if any(folder.glob("*.jpg")):
                return folder.name, folder
    return None, None


def task_b_false_acceptance(student_id, student_dir):
    print("\n" + "=" * 70)
    print("TASK B: FALSE ACCEPTANCE DIAGNOSIS")
    print("=" * 70)
    
    enrollment_service = EnrollmentService()
    gallery = enrollment_service.load_active_gallery()
    recognizer = FaceRecognizer()  # Uses config threshold=0.6, margin=0.1
    embedder = FaceEmbedder()
    detector = FaceDetector()
    
    print(f"\nGallery: {len(gallery)} samples, {len(set(id for id, _ in gallery))} distinct identities")
    print(f"Threshold: {recognizer.threshold}, Margin: {recognizer.margin}")
    
    # Check for available test images
    if student_id is None:
        print("\nWARNING: No enrolled student with photos found in data/enrollment_images.")
        print("Enroll a student through the Student Registry, then run this again.")
        return
    print(f"Known test person: student {student_id} ({student_dir})")

    # Test 1: Known enrolled person using existing enrolled image
    print("\n--- Test 1: Known enrolled person (existing enrollment image) ---")
    if student_dir.exists():
        images = list(student_dir.glob("*.jpg"))
        if images:
            test_img = cv2.imread(str(images[0]))
            if test_img is not None:
                print(f"Test image: {images[0].name}")
                faces = detector.detect_faces(test_img)
                if faces:
                    x, y, w, h = faces[0].bbox
                    crop = test_img[y:y+h, x:x+w]
                    query_emb = embedder.generate_embedding(crop)
                    if query_emb:
                        result = recognizer.recognize(query_emb, gallery)
                        print(f"  Person: Enrolled ({student_id})")
                        print(f"  Expected: {student_id}")
                        print(f"  Predicted: {result.identity}")
                        print(f"  Best distance: {result.distance:.4f}")
                        print(f"  Second-best distance: {result.second_best_distance:.4f}")
                        print(f"  Threshold check: {'PASS' if result.distance <= recognizer.threshold else 'FAIL'}")
                        if result.second_best_distance != float('inf'):
                            margin_gap = result.second_best_distance - result.distance
                            print(f"  Margin check: {'PASS' if margin_gap >= recognizer.margin else 'FAIL'} (gap={margin_gap:.4f})")
                        print(f"  Result: {result.reason}, is_match={result.is_match}")
                    else:
                        print("  ERROR: Could not generate embedding")
                else:
                    print("  ERROR: No face detected in test image")
            else:
                print("  ERROR: Could not read image")
        else:
            print(f"  No images found for student {student_id}")
    else:
        print(f"  Student {student_id} enrollment_images directory not found")
    
    # Test 2: Different enrolled angle (if multiple samples exist)
    print("\n--- Test 2: Known enrolled person (different angle if available) ---")
    if student_dir.exists():
        images = list(student_dir.glob("*.jpg"))
        if len(images) > 1:
            test_img = cv2.imread(str(images[1]))
            if test_img is not None:
                print(f"Test image: {images[1].name}")
                faces = detector.detect_faces(test_img)
                if faces:
                    x, y, w, h = faces[0].bbox
                    crop = test_img[y:y+h, x:x+w]
                    query_emb = embedder.generate_embedding(crop)
                    if query_emb:
                        result = recognizer.recognize(query_emb, gallery)
                        print(f"  Person: Enrolled ({student_id})")
                        print(f"  Expected: {student_id}")
                        print(f"  Predicted: {result.identity}")
                        print(f"  Best distance: {result.distance:.4f}")
                        print(f"  Second-best distance: {result.second_best_distance:.4f}")
                        print(f"  Threshold check: {'PASS' if result.distance <= recognizer.threshold else 'FAIL'}")
                        if result.second_best_distance != float('inf'):
                            margin_gap = result.second_best_distance - result.distance
                            print(f"  Margin check: {'PASS' if margin_gap >= recognizer.margin else 'FAIL'} (gap={margin_gap:.4f})")
                        print(f"  Result: {result.reason}, is_match={result.is_match}")
        else:
            print("  Only one image available; skipping different-angle test")
    
    # Test 3: Unregistered people
    print("\n--- Test 3: Unregistered people (if available) ---")
    unknown_dir = project_root / "data" / "unknown_faces"
    if unknown_dir.exists():
        unknown_images = list(unknown_dir.glob("*.jpg"))[:2]  # Up to 2
        if unknown_images:
            for idx, img_path in enumerate(unknown_images, 1):
                test_img = cv2.imread(str(img_path))
                if test_img is not None:
                    print(f"\n  Unknown person #{idx}: {img_path.name}")
                    faces = detector.detect_faces(test_img)
                    if faces:
                        x, y, w, h = faces[0].bbox
                        crop = test_img[y:y+h, x:x+w]
                        query_emb = embedder.generate_embedding(crop)
                        if query_emb:
                            result = recognizer.recognize(query_emb, gallery)
                            print(f"    Person: Unknown (not enrolled)")
                            print(f"    Expected: Unknown")
                            print(f"    Predicted: {result.identity}")
                            print(f"    Best distance: {result.distance:.4f}")
                            print(f"    Second-best distance: {result.second_best_distance:.4f}")
                            print(f"    Threshold check: {'PASS' if result.distance <= recognizer.threshold else 'FAIL'}")
                            if result.second_best_distance != float('inf'):
                                margin_gap = result.second_best_distance - result.distance
                                print(f"    Margin check: {'PASS' if margin_gap >= recognizer.margin else 'FAIL'} (gap={margin_gap:.4f})")
                            print(f"    Result: {result.reason}, is_match={result.is_match}")
                            if result.is_match:
                                print(f"    **FALSE ACCEPTANCE: Unknown person matched as {result.identity}**")
        else:
            print("  No unknown_faces images available")
    else:
        print("  No unknown_faces directory; cannot test with unregistered people")
    
    # Test 4: Single-identity gallery (isolated fixture)
    print("\n--- Test 4: Single-identity gallery (isolated in-memory test) ---")
    # One sample of the test student, so the query below really is the same person
    single_gallery = [s for s in gallery if s[0] == student_id][:1]
    if single_gallery:
        print(f"  Single-identity gallery: 1 sample of {single_gallery[0][0]}")
        # Use the same enrolled image as query
        if student_dir.exists():
            images = list(student_dir.glob("*.jpg"))
            if images:
                test_img = cv2.imread(str(images[0]))
                if test_img is not None:
                    faces = detector.detect_faces(test_img)
                    if faces:
                        x, y, w, h = faces[0].bbox
                        crop = test_img[y:y+h, x:x+w]
                        query_emb = embedder.generate_embedding(crop)
                        if query_emb:
                            result = recognizer.recognize(query_emb, single_gallery)
                            print(f"  Query: Same enrolled person")
                            print(f"  Predicted: {result.identity}")
                            print(f"  Best distance: {result.distance:.4f}")
                            print(f"  Second-best distance: {result.second_best_distance:.4f} (should be inf)")
                            print(f"  Result: {result.reason}, is_match={result.is_match}")
                            print(f"  Observation: With only 1 identity, second-best is inf, so ambiguity check is skipped")

def task_c_performance(student_dir):
    print("\n" + "=" * 70)
    print("TASK C: PERFORMANCE MEASUREMENT")
    print("=" * 70)
    
    enrollment_service = EnrollmentService()
    gallery = enrollment_service.load_active_gallery()
    recognizer = FaceRecognizer()
    embedder = FaceEmbedder()
    detector = FaceDetector()
    
    print(f"\nGallery size: {len(gallery)} samples")
    print(f"Distinct identities: {len(set(id for id, _ in gallery))}")
    
    # Find a test image
    test_img_path = student_dir
    if test_img_path is not None and test_img_path.exists():
        images = list(test_img_path.glob("*.jpg"))
        if images:
            test_img = cv2.imread(str(images[0]))
            if test_img is not None:
                print(f"Test image: {images[0].name}")
                
                # Measure performance over multiple runs
                n_runs = 10
                detection_times = []
                embedding_times = []
                recognition_times = []
                total_times = []
                
                for _ in range(n_runs):
                    t0 = time.perf_counter()
                    
                    # Detection
                    t_det_start = time.perf_counter()
                    faces = detector.detect_faces(test_img)
                    t_det_end = time.perf_counter()
                    detection_times.append((t_det_end - t_det_start) * 1000)
                    
                    if faces:
                        x, y, w, h = faces[0].bbox
                        crop = test_img[y:y+h, x:x+w]
                        
                        # Embedding
                        t_emb_start = time.perf_counter()
                        query_emb = embedder.generate_embedding(crop)
                        t_emb_end = time.perf_counter()
                        embedding_times.append((t_emb_end - t_emb_start) * 1000)
                        
                        if query_emb:
                            # Recognition
                            t_rec_start = time.perf_counter()
                            result = recognizer.recognize(query_emb, gallery)
                            t_rec_end = time.perf_counter()
                            recognition_times.append((t_rec_end - t_rec_start) * 1000)
                    
                    t1 = time.perf_counter()
                    total_times.append((t1 - t0) * 1000)
                
                # Compute statistics
                def stats(times):
                    if not times:
                        return 0, 0
                    times_sorted = sorted(times)
                    median = times_sorted[len(times_sorted) // 2]
                    p95_idx = int(len(times_sorted) * 0.95)
                    p95 = times_sorted[min(p95_idx, len(times_sorted) - 1)]
                    return median, p95
                
                print(f"\nPerformance over {n_runs} runs (milliseconds):")
                print(f"                    Median    P95")
                det_med, det_p95 = stats(detection_times)
                print(f"  Face detection:   {det_med:6.1f}  {det_p95:6.1f}")
                emb_med, emb_p95 = stats(embedding_times)
                print(f"  Embedding gen:    {emb_med:6.1f}  {emb_p95:6.1f}")
                rec_med, rec_p95 = stats(recognition_times)
                print(f"  Recognition:      {rec_med:6.1f}  {rec_p95:6.1f}")
                tot_med, tot_p95 = stats(total_times)
                print(f"  Total per frame:  {tot_med:6.1f}  {tot_p95:6.1f}")
                
                print(f"\nObservations:")
                print(f"  - Gallery has {len(gallery)} samples from "
                      f"{len(set(i for i, _ in gallery))} identities")
                print(f"  - Recognition computes {len(gallery)} Euclidean distances per query")
                print(f"  - At 6 fps CV processing rate, budget is ~166ms per frame")
                if tot_med > 166:
                    print(f"  - WARNING: Median total time ({tot_med:.1f}ms) exceeds 6fps budget")
                else:
                    print(f"  - Total time is within 6fps processing budget")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Recognition and performance diagnostics")
    parser.add_argument("--student-id", help="enrolled student to use as the known person "
                                             "(default: first student with enrollment photos)")
    args = parser.parse_args()

    test_student_id, test_student_dir = pick_test_student(args.student_id)
    task_b_false_acceptance(test_student_id, test_student_dir)
    task_c_performance(test_student_dir)
    
    print("\n" + "=" * 70)
    print("Tasks B and C complete.")
    print("=" * 70)
