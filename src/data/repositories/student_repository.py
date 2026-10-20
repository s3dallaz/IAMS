"""
Student Repository for database persistence operations.

Provides clean database access for the students table without business logic.
"""

import sqlite3
from typing import Optional, List, Dict, Any
from pathlib import Path

from src.database.connection import get_db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StudentRepository:
    """
    Repository for student persistence operations.
    
    Handles database access for the students table.
    Does not contain business logic.
    """
    
    def __init__(self, database_path: Optional[Path] = None):
        """
        Initialize the student repository.
        
        Args:
            database_path: Path to database file. If None, uses config.DATABASE_PATH
        """
        self.database_path = database_path
    
    def create_student(
        self,
        student_id: str,
        name: str,
        email: Optional[str] = None,
        department: Optional[str] = None,
        notes: Optional[str] = None
    ) -> int:
        """
        Create a new student in the database.
        
        Args:
            student_id: Unique student identifier (e.g., "2021001")
            name: Student's full name
            email: Student's email address (optional)
            department: Student's department (optional)
            notes: Additional notes (optional)
        
        Returns:
            Database ID of the newly created student
        
        Raises:
            sqlite3.IntegrityError: If student_id already exists
            sqlite3.Error: On other database errors
        """
        query = """
            INSERT INTO students (student_id, name, email, department, notes)
            VALUES (?, ?, ?, ?, ?)
        """
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (student_id, name, email, department, notes))
                student_db_id = cursor.lastrowid
                logger.info(f"Created student: {student_id} (DB ID: {student_db_id})")
                return student_db_id
        except sqlite3.IntegrityError as e:
            logger.error(f"Failed to create student {student_id}: {e}")
            raise
        except sqlite3.Error as e:
            logger.error(f"Database error creating student: {e}")
            raise
    
    def get_by_id(self, student_db_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a student by database ID.
        
        Args:
            student_db_id: Database primary key
        
        Returns:
            Student record as dictionary, or None if not found
        """
        query = "SELECT * FROM students WHERE id = ?"
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (student_db_id,))
                row = cursor.fetchone()
                
                if row:
                    return dict(row)
                return None
        except sqlite3.Error as e:
            logger.error(f"Database error getting student by id {student_db_id}: {e}")
            raise
    
    def get_by_student_id(self, student_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a student by student ID.
        
        Args:
            student_id: Unique student identifier (e.g., "2021001")
        
        Returns:
            Student record as dictionary, or None if not found
        """
        query = "SELECT * FROM students WHERE student_id = ?"
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (student_id,))
                row = cursor.fetchone()
                
                if row:
                    return dict(row)
                return None
        except sqlite3.Error as e:
            logger.error(f"Database error getting student by student_id {student_id}: {e}")
            raise
    
    def get_all(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get all students.
        
        Args:
            active_only: If True, return only active students (is_active = 1)
        
        Returns:
            List of student records as dictionaries
        """
        if active_only:
            query = "SELECT * FROM students WHERE is_active = 1 ORDER BY student_id"
        else:
            query = "SELECT * FROM students ORDER BY student_id"
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Database error getting all students: {e}")
            raise
    
    def update_student(
        self,
        student_db_id: int,
        student_id: Optional[str] = None,
        name: Optional[str] = None,
        email: Optional[str] = None,
        department: Optional[str] = None,
        notes: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> int:
        """
        Update a student's information.
        
        Args:
            student_db_id: Database primary key
            student_id: New student ID (optional)
            name: New name (optional)
            email: New email (optional)
            department: New department (optional)
            notes: New notes (optional)
            is_active: New active status (optional)
        
        Returns:
            Number of rows affected (0 if student not found, 1 if updated)
        
        Raises:
            sqlite3.IntegrityError: If new student_id already exists
            sqlite3.Error: On other database errors
        """
        # Build dynamic UPDATE query based on provided fields
        updates = []
        params = []
        
        if student_id is not None:
            updates.append("student_id = ?")
            params.append(student_id)
        
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        
        if email is not None:
            updates.append("email = ?")
            params.append(email)
        
        if department is not None:
            updates.append("department = ?")
            params.append(department)
        
        if notes is not None:
            updates.append("notes = ?")
            params.append(notes)
        
        if is_active is not None:
            updates.append("is_active = ?")
            params.append(1 if is_active else 0)
        
        if not updates:
            # No fields to update
            logger.warning(f"No fields provided to update for student {student_db_id}")
            return 0
        
        query = f"UPDATE students SET {', '.join(updates)} WHERE id = ?"
        params.append(student_db_id)
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows_affected = cursor.rowcount
                
                if rows_affected > 0:
                    logger.info(f"Updated student DB ID {student_db_id}")
                else:
                    logger.warning(f"Student DB ID {student_db_id} not found for update")
                
                return rows_affected
        except sqlite3.IntegrityError as e:
            logger.error(f"Failed to update student {student_db_id}: {e}")
            raise
        except sqlite3.Error as e:
            logger.error(f"Database error updating student: {e}")
            raise
    
    def deactivate_student(self, student_db_id: int) -> int:
        """
        Deactivate a student (set is_active = 0).
        
        Args:
            student_db_id: Database primary key
        
        Returns:
            Number of rows affected (0 if student not found, 1 if deactivated)
        """
        query = "UPDATE students SET is_active = 0 WHERE id = ?"
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (student_db_id,))
                rows_affected = cursor.rowcount
                
                if rows_affected > 0:
                    logger.info(f"Deactivated student DB ID {student_db_id}")
                else:
                    logger.warning(f"Student DB ID {student_db_id} not found for deactivation")
                
                return rows_affected
        except sqlite3.Error as e:
            logger.error(f"Database error deactivating student: {e}")
            raise
    
    def activate_student(self, student_db_id: int) -> int:
        """
        Activate a student (set is_active = 1).
        
        Args:
            student_db_id: Database primary key
        
        Returns:
            Number of rows affected (0 if student not found, 1 if activated)
        """
        query = "UPDATE students SET is_active = 1 WHERE id = ?"
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (student_db_id,))
                rows_affected = cursor.rowcount
                
                if rows_affected > 0:
                    logger.info(f"Activated student DB ID {student_db_id}")
                else:
                    logger.warning(f"Student DB ID {student_db_id} not found for activation")
                
                return rows_affected
        except sqlite3.Error as e:
            logger.error(f"Database error activating student: {e}")
            raise
    
    def exists_by_student_id(self, student_id: str) -> bool:
        """
        Check if a student exists by student ID.
        
        Args:
            student_id: Unique student identifier (e.g., "2021001")
        
        Returns:
            True if student exists, False otherwise
        """
        query = "SELECT 1 FROM students WHERE student_id = ? LIMIT 1"
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (student_id,))
                return cursor.fetchone() is not None
        except sqlite3.Error as e:
            logger.error(f"Database error checking student existence: {e}")
            raise

    def delete_student(self, student_db_id: int) -> int:
        """
        Delete a student by database ID.
        
        Foreign key ON DELETE CASCADE will clean up enrollments and attendance records.
        
        Args:
            student_db_id: Database primary key
            
        Returns:
            Number of rows affected (1 if deleted, 0 if not found)
        """
        query = "DELETE FROM students WHERE id = ?"
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (student_db_id,))
                rows_affected = cursor.rowcount
                if rows_affected > 0:
                    logger.info(f"Deleted student DB ID {student_db_id}")
                else:
                    logger.warning(f"Student DB ID {student_db_id} not found for deletion")
                return rows_affected
        except sqlite3.Error as e:
            logger.error(f"Database error deleting student {student_db_id}: {e}")
            raise

    def bulk_save(self, creates: List[Dict[str, Any]], updates: List[Dict[str, Any]]) -> None:
        """
        Insert and update many students in ONE transaction (all or nothing).

        Args:
            creates: dicts with student_id, name, email, department, notes
            updates: the same keys; the row with that student_id is updated
        """
        insert = """
            INSERT INTO students (student_id, name, email, department, notes)
            VALUES (:student_id, :name, :email, :department, :notes)
        """
        update = """
            UPDATE students SET name = :name, email = :email, department = :department, notes = :notes
            WHERE student_id = :student_id
        """
        columns = ("student_id", "name", "email", "department", "notes")
        try:
            with get_db_connection(self.database_path) as conn:
                conn.executemany(insert, [{k: row.get(k) for k in columns} for row in creates])
                conn.executemany(update, [{k: row.get(k) for k in columns} for row in updates])
            logger.info(f"Bulk saved students: {len(creates)} created, {len(updates)} updated")
        except sqlite3.Error as e:
            logger.error(f"Database error in bulk student save (nothing was saved): {e}")
            raise


if __name__ == "__main__":
    # Simple manual test
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.database.schema import initialize_database
    import tempfile
    
    print("Testing StudentRepository...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        test_db = Path(f.name)
    
    try:
        # Initialize schema
        initialize_database(test_db)
        
        # Create repository
        repo = StudentRepository(test_db)
        
        # Test create
        student_db_id = repo.create_student(
            student_id="2021001",
            name="John Doe",
            email="john@example.com",
            department="Computer Science"
        )
        print(f"[OK] Created student with DB ID: {student_db_id}")
        
        # Test get by id
        student = repo.get_by_id(student_db_id)
        print(f"[OK] Retrieved student by DB ID: {student['name']}")
        
        # Test get by student_id
        student = repo.get_by_student_id("2021001")
        print(f"[OK] Retrieved student by student_id: {student['name']}")
        
        # Test exists
        exists = repo.exists_by_student_id("2021001")
        print(f"[OK] Student exists: {exists}")
        
        # Test get all
        all_students = repo.get_all()
        print(f"[OK] Total students: {len(all_students)}")
        
        # Test update
        rows = repo.update_student(student_db_id, name="Jane Doe")
        print(f"[OK] Updated {rows} student(s)")
        
        # Test deactivate
        rows = repo.deactivate_student(student_db_id)
        print(f"[OK] Deactivated {rows} student(s)")
        
        # Test activate
        rows = repo.activate_student(student_db_id)
        print(f"[OK] Activated {rows} student(s)")
        
        print("\n[OK] StudentRepository tests passed")
        
    finally:
        # Cleanup
        if test_db.exists():
            test_db.unlink()
