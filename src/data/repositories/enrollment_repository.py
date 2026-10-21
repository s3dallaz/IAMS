"""
Enrollment Repository for face encoding persistence operations.

Provides clean database access for the enrollments table without business logic.
"""

import sqlite3
from typing import Optional, List, Dict, Any
from pathlib import Path

from src.database.connection import get_db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnrollmentRepository:
    """
    Repository for student face enrollment persistence operations.
    
    Handles database access for the enrollments table.
    Does not contain business logic.
    """
    
    def __init__(self, database_path: Optional[Path] = None):
        """
        Initialize the enrollment repository.
        
        Args:
            database_path: Path to database file. If None, uses config.DATABASE_PATH
        """
        self.database_path = database_path
    
    def create_enrollment(
        self,
        student_id: int,
        encoding_path: str,
        enrollment_method: str = 'camera',
        image_count: int = 1,
        quality_score: Optional[float] = None,
        view_angle: str = 'unknown',
        is_active: bool = True
    ) -> int:
        """
        Create a new enrollment record.
        
        Args:
            student_id: Foreign key to students.id
            encoding_path: Path to the saved .npy or image file containing face encoding
            enrollment_method: 'camera' or 'upload'
            image_count: Number of sample images used
            quality_score: Optional embedding quality / confidence score
            view_angle: Angle of the face sample ('frontal', 'left', 'right', 'unknown')
            is_active: Whether this enrollment is active
            
        Returns:
            Database primary key of the new enrollment record
        """
        query = """
            INSERT INTO enrollments (
                student_id, encoding_path, enrollment_method,
                image_count, quality_score, view_angle, is_active
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    query,
                    (student_id, encoding_path, enrollment_method, image_count, quality_score, view_angle, 1 if is_active else 0)
                )
                enrollment_id = cursor.lastrowid
                logger.info(f"Created enrollment for student DB ID {student_id} (Enrollment ID: {enrollment_id})")
                return enrollment_id
        except sqlite3.Error as e:
            logger.error(f"Database error creating enrollment: {e}")
            raise
    
    def get_by_id(self, enrollment_id: int) -> Optional[Dict[str, Any]]:
        """Get an enrollment record by its ID."""
        query = "SELECT * FROM enrollments WHERE id = ?"
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (enrollment_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except sqlite3.Error as e:
            logger.error(f"Database error getting enrollment by id {enrollment_id}: {e}")
            raise
    
    def get_by_student_id(self, student_id: int, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get all enrollment records for a student."""
        if active_only:
            query = "SELECT * FROM enrollments WHERE student_id = ? AND is_active = 1 ORDER BY enrolled_at DESC"
        else:
            query = "SELECT * FROM enrollments WHERE student_id = ? ORDER BY enrolled_at DESC"
            
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (student_id,))
                rows = cursor.fetchall()
                return [dict(r) for r in rows]
        except sqlite3.Error as e:
            logger.error(f"Database error getting enrollments for student {student_id}: {e}")
            raise
    
    def get_all(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get all enrollment records."""
        if active_only:
            query = "SELECT * FROM enrollments WHERE is_active = 1 ORDER BY enrolled_at DESC"
        else:
            query = "SELECT * FROM enrollments ORDER BY enrolled_at DESC"
            
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()
                return [dict(r) for r in rows]
        except sqlite3.Error as e:
            logger.error(f"Database error getting all enrollments: {e}")
            raise
    
    def get_active_gallery_keys(self) -> List[tuple]:
        """
        (enrollment id, encoding path, student_id) of every active sample of
        an active student, ordered by id. Cheap compared to loading the
        encodings: used to notice when the recognition gallery changed.
        """
        query = """
            SELECT e.id, e.encoding_path, s.student_id
            FROM enrollments e JOIN students s ON s.id = e.student_id
            WHERE e.is_active = 1 AND s.is_active = 1
            ORDER BY e.id
        """
        try:
            with get_db_connection(self.database_path) as conn:
                return [tuple(r) for r in conn.execute(query).fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Database error getting active gallery keys: {e}")
            raise

    def get_active_gallery_fingerprint(self) -> tuple:
        """
        (count, max id, sum of ids, sum of student DB ids, sum of student_id
        lengths) of the active gallery samples: changes whenever a sample is
        added, removed or (de)activated, or a student is (de)activated or gets
        another student_id. ~25 ms for 65,000 samples, against ~160 ms for
        get_active_gallery_keys() (which also reads every encoding path).
        """
        query = """
            SELECT COUNT(*), COALESCE(MAX(e.id), 0), COALESCE(SUM(e.id), 0),
                   COALESCE(SUM(s.id), 0), COALESCE(SUM(LENGTH(s.student_id)), 0)
            FROM enrollments e JOIN students s ON s.id = e.student_id
            WHERE e.is_active = 1 AND s.is_active = 1
        """
        try:
            with get_db_connection(self.database_path) as conn:
                return tuple(conn.execute(query).fetchone())
        except sqlite3.Error as e:
            logger.error(f"Database error getting the active gallery fingerprint: {e}")
            raise

    def delete_by_student_id(self, student_id: int) -> int:
        """Delete all enrollments for a student."""
        query = "DELETE FROM enrollments WHERE student_id = ?"
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (student_id,))
                return cursor.rowcount
        except sqlite3.Error as e:
            logger.error(f"Database error deleting enrollments for student {student_id}: {e}")
            raise
