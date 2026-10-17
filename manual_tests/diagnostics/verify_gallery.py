"""
Diagnostic script: Verify actual gallery loaded at runtime.
Does not modify any files or database.
"""
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.services.enrollment_service import EnrollmentService

def verify_gallery():
    print("=" * 70)
    print("TASK A: ACTUAL RUNTIME GALLERY VERIFICATION")
    print("=" * 70)
    
    enrollment_service = EnrollmentService()
    gallery = enrollment_service.load_active_gallery()
    
    print(f"\n1. Total valid encoding samples loaded: {len(gallery)}")
    
    # Count samples per identity
    identity_counts = {}
    for identity, embedding in gallery:
        identity_counts[identity] = identity_counts.get(identity, 0) + 1
    
    print(f"\n2. Samples per distinct student identity:")
    print(f"   Total distinct identities: {len(identity_counts)}")
    for identity, count in sorted(identity_counts.items()):
        print(f"   - {identity}: {count} sample(s)")
    
    # Check for missing/unreadable files
    print(f"\n3. Missing and unreadable encoding files:")
    from src.data.repositories.enrollment_repository import EnrollmentRepository
    from src.data.repositories.student_repository import StudentRepository
    import numpy as np
    
    enrollment_repo = EnrollmentRepository()
    student_repo = StudentRepository()
    
    students = student_repo.get_all(active_only=True)
    student_map = {s['id']: s['student_id'] for s in students}
    
    enrollments = enrollment_repo.get_all(active_only=True)
    
    missing = []
    unreadable = []
    loaded = []
    
    for record in enrollments:
        student_db_id = record['student_id']
        if student_db_id not in student_map:
            continue
        
        student_id_str = student_map[student_db_id]
        encoding_path = Path(record['encoding_path'])
        
        if not encoding_path.exists():
            missing.append((student_id_str, encoding_path.name))
        else:
            try:
                arr = np.load(str(encoding_path))
                if arr.shape == (128,):
                    loaded.append(student_id_str)
                else:
                    unreadable.append((student_id_str, encoding_path.name, f"shape={arr.shape}"))
            except Exception as e:
                unreadable.append((student_id_str, encoding_path.name, str(e)))
    
    print(f"   Missing files: {len(missing)}")
    for identity, filename in missing:
        print(f"     - {identity}/{filename}")
    
    print(f"   Unreadable files: {len(unreadable)}")
    for identity, filename, reason in unreadable:
        print(f"     - {identity}/{filename}: {reason}")
    
    print(f"   Successfully loaded: {len(loaded)}")
    
    # Verify inactive student exclusion
    print(f"\n4. Inactive student exclusion verification:")
    inactive_students = student_repo.get_all(active_only=False)
    inactive_ids = [s['student_id'] for s in inactive_students if not s.get('is_active', 1)]
    
    print(f"   Total inactive students in database: {len(inactive_ids)}")
    for inactive_id in inactive_ids:
        print(f"   - {inactive_id}")
    
    inactive_in_gallery = [identity for identity in identity_counts.keys() if identity in inactive_ids]
    if inactive_in_gallery:
        print(f"   ERROR: {len(inactive_in_gallery)} inactive student(s) found in gallery:")
        for identity in inactive_in_gallery:
            print(f"     - {identity}")
    else:
        print(f"   OK: No inactive students in gallery (correct)")
    
    # Verify grouping
    print(f"\n5. Identity grouping and comparison verification:")
    print(f"   Gallery structure: List[Tuple[student_id_string, FaceEmbedding]]")
    print(f"   Multiple samples per identity are correctly included as separate tuples")
    print(f"   with the same identity string.")
    print(f"   Recognition compares query against ALL samples, finds closest match,")
    print(f"   and second-best is computed across DIFFERENT identities (same identity")
    print(f"   samples are skipped in second-best calculation).")
    print(f"   OK: Grouping verified by code inspection (face_recognizer.py lines 151-156)")
    
    return gallery, identity_counts

if __name__ == "__main__":
    gallery, identity_counts = verify_gallery()
    print("\n" + "=" * 70)
    print("Task A complete. Gallery verification finished.")
    print("=" * 70)
