"""
Student Service layer for business logic and validation.

Provides application-level student management operations with validation,
normalization, and business rule enforcement.
"""

import re
import sqlite3
from typing import Optional, List, Dict, Any
from pathlib import Path

from src.data.repositories.student_repository import StudentRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Service-level Exceptions
# ============================================================================

class StudentServiceError(Exception):
    """Base exception for student service errors."""
    pass


class InvalidStudentDataError(StudentServiceError):
    """Raised when student data fails validation."""
    pass


class DuplicateStudentError(StudentServiceError):
    """Raised when attempting to create a student with duplicate student_id."""
    pass


class StudentNotFoundError(StudentServiceError):
    """Raised when a requested student does not exist."""
    pass


# ============================================================================
# Student Service
# ============================================================================

class StudentService:
    """
    Service layer for student management.
    
    Handles validation, normalization, business rules, and coordination
    with StudentRepository. Does not contain SQL or database connection logic.
    """
    
    # Email validation regex (basic format check)
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    def __init__(self, database_path: Optional[Path] = None):
        """
        Initialize the student service.
        
        Args:
            database_path: Path to database file. If None, uses config.DATABASE_PATH
        """
        self.repository = StudentRepository(database_path)
        logger.debug("StudentService initialized")
    
    def create_student(
        self,
        student_id: str,
        name: str,
        email: Optional[str] = None,
        department: Optional[str] = None,
        notes: Optional[str] = None
    ) -> int:
        """
        Create a new student with validation.
        
        Args:
            student_id: Unique student identifier (e.g., "2021001")
            name: Student's full name
            email: Student's email address (optional)
            department: Student's department (optional)
            notes: Additional notes (optional)
        
        Returns:
            Database ID of the newly created student
        
        Raises:
            InvalidStudentDataError: If validation fails
            DuplicateStudentError: If student_id already exists
        """
        # Validate and normalize student_id
        if not isinstance(student_id, str):
            raise InvalidStudentDataError("student_id must be a string")
        
        student_id = student_id.strip()
        if not student_id:
            raise InvalidStudentDataError("student_id cannot be empty")
        
        # Validate and normalize name
        if not isinstance(name, str):
            raise InvalidStudentDataError("name must be a string")
        
        name = name.strip()
        if not name:
            raise InvalidStudentDataError("name cannot be empty")
        
        # Validate and normalize email if provided
        if email is not None:
            if not isinstance(email, str):
                raise InvalidStudentDataError("email must be a string")
            
            email = email.strip()
            if email and not self.EMAIL_PATTERN.match(email):
                raise InvalidStudentDataError("email format is invalid")
            
            # Convert empty string to None for database
            if not email:
                email = None
        
        # Validate department if provided
        if department is not None:
            if not isinstance(department, str):
                raise InvalidStudentDataError("department must be a string")
            department = department.strip()
            # Convert empty string to None for database
            if not department:
                department = None
        
        # Validate notes if provided
        if notes is not None:
            if not isinstance(notes, str):
                raise InvalidStudentDataError("notes must be a string")
            notes = notes.strip()
            # Convert empty string to None for database
            if not notes:
                notes = None
        
        # Check for duplicate student_id
        if self.repository.exists_by_student_id(student_id):
            logger.warning(f"Attempted to create duplicate student_id: {student_id}")
            raise DuplicateStudentError(f"Student with ID '{student_id}' already exists")
        
        # Create student through repository
        try:
            student_db_id = self.repository.create_student(
                student_id=student_id,
                name=name,
                email=email,
                department=department,
                notes=notes
            )
            
            # Post-insert verification: Confirm record was written and committed to SQLite
            saved_student = self.repository.get_by_id(student_db_id)
            if not saved_student or saved_student['student_id'] != student_id:
                raise StudentServiceError(f"Database insertion verification failed for student ID '{student_id}'.")
                
            logger.info(f"Student created and verified successfully: {student_id} (DB ID: {student_db_id})")
            return student_db_id
            
        except sqlite3.IntegrityError as e:
            # Catch race condition where student was created between check and insert
            logger.error(f"Integrity error creating student: {e}")
            raise DuplicateStudentError(f"Student with ID '{student_id}' already exists")
    
    def get_student_by_id(self, student_db_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a student by database ID.
        
        Args:
            student_db_id: Database primary key
        
        Returns:
            Student record as dictionary, or None if not found
        
        Raises:
            InvalidStudentDataError: If student_db_id is invalid
        """
        if not isinstance(student_db_id, int) or student_db_id <= 0:
            raise InvalidStudentDataError("student_db_id must be a positive integer")
        
        return self.repository.get_by_id(student_db_id)
    
    def get_student_by_student_id(self, student_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a student by student ID.
        
        Args:
            student_id: Unique student identifier (e.g., "2021001")
        
        Returns:
            Student record as dictionary, or None if not found
        
        Raises:
            InvalidStudentDataError: If student_id is invalid
        """
        if not isinstance(student_id, str):
            raise InvalidStudentDataError("student_id must be a string")
        
        student_id = student_id.strip()
        if not student_id:
            raise InvalidStudentDataError("student_id cannot be empty")
        
        return self.repository.get_by_student_id(student_id)
    
    def get_all_students(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get all students.
        
        Args:
            active_only: If True, return only active students (is_active = 1)
        
        Returns:
            List of student records as dictionaries
        """
        return self.repository.get_all(active_only=active_only)
    
    def search_students(self, text: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Students whose name or student ID contains `text`, at most `limit`
        (keeps pickers light when thousands of students are registered).
        """
        text = (text or "").strip()
        if not text:
            return []
        return self.repository.search(text, limit=limit)
    
    def student_exists(self, student_id: str) -> bool:
        """
        Check if a student exists by student ID.
        
        Args:
            student_id: Unique student identifier (e.g., "2021001")
        
        Returns:
            True if student exists, False otherwise
        
        Raises:
            InvalidStudentDataError: If student_id is invalid
        """
        if not isinstance(student_id, str):
            raise InvalidStudentDataError("student_id must be a string")
        
        student_id = student_id.strip()
        if not student_id:
            raise InvalidStudentDataError("student_id cannot be empty")
        
        return self.repository.exists_by_student_id(student_id)
    
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
            Number of rows affected (1 if updated, 0 if student not found)
        
        Raises:
            InvalidStudentDataError: If validation fails
            StudentNotFoundError: If student does not exist
            DuplicateStudentError: If new student_id already exists
        """
        # Validate student_db_id
        if not isinstance(student_db_id, int) or student_db_id <= 0:
            raise InvalidStudentDataError("student_db_id must be a positive integer")
        
        # Verify student exists
        existing_student = self.repository.get_by_id(student_db_id)
        if not existing_student:
            raise StudentNotFoundError(f"Student with DB ID {student_db_id} not found")
        
        # Validate and normalize student_id if provided
        if student_id is not None:
            if not isinstance(student_id, str):
                raise InvalidStudentDataError("student_id must be a string")
            
            student_id = student_id.strip()
            if not student_id:
                raise InvalidStudentDataError("student_id cannot be empty")
            
            # Check if new student_id is already taken by another student
            if student_id != existing_student['student_id']:
                if self.repository.exists_by_student_id(student_id):
                    raise DuplicateStudentError(f"Student with ID '{student_id}' already exists")
        
        # Validate and normalize name if provided
        if name is not None:
            if not isinstance(name, str):
                raise InvalidStudentDataError("name must be a string")
            
            name = name.strip()
            if not name:
                raise InvalidStudentDataError("name cannot be empty")
        
        # Validate and normalize email if provided
        if email is not None:
            if not isinstance(email, str):
                raise InvalidStudentDataError("email must be a string")
            
            email = email.strip()
            if email and not self.EMAIL_PATTERN.match(email):
                raise InvalidStudentDataError("email format is invalid")
            
            # Convert empty string to None for database
            if not email:
                email = None
        
        # Validate department if provided
        if department is not None:
            if not isinstance(department, str):
                raise InvalidStudentDataError("department must be a string")
            department = department.strip()
            # Convert empty string to None for database
            if not department:
                department = None
        
        # Validate notes if provided
        if notes is not None:
            if not isinstance(notes, str):
                raise InvalidStudentDataError("notes must be a string")
            notes = notes.strip()
            # Convert empty string to None for database
            if not notes:
                notes = None
        
        # Validate is_active if provided
        if is_active is not None:
            if not isinstance(is_active, bool):
                raise InvalidStudentDataError("is_active must be a boolean")
        
        # Update through repository
        try:
            rows_affected = self.repository.update_student(
                student_db_id=student_db_id,
                student_id=student_id,
                name=name,
                email=email,
                department=department,
                notes=notes,
                is_active=is_active
            )
            
            if rows_affected > 0:
                logger.info(f"Student DB ID {student_db_id} updated successfully")
            
            return rows_affected
            
        except sqlite3.IntegrityError as e:
            # Catch race condition where student_id was taken between check and update
            logger.error(f"Integrity error updating student: {e}")
            raise DuplicateStudentError(f"Student with ID '{student_id}' already exists")
    
    def activate_student(self, student_db_id: int) -> int:
        """
        Activate a student (set is_active = 1).
        
        Args:
            student_db_id: Database primary key
        
        Returns:
            Number of rows affected (1 if activated, 0 if student not found)
        
        Raises:
            InvalidStudentDataError: If student_db_id is invalid
            StudentNotFoundError: If student does not exist
        """
        # Validate student_db_id
        if not isinstance(student_db_id, int) or student_db_id <= 0:
            raise InvalidStudentDataError("student_db_id must be a positive integer")
        
        # Verify student exists
        existing_student = self.repository.get_by_id(student_db_id)
        if not existing_student:
            raise StudentNotFoundError(f"Student with DB ID {student_db_id} not found")
        
        rows_affected = self.repository.activate_student(student_db_id)
        
        if rows_affected > 0:
            logger.info(f"Student DB ID {student_db_id} activated successfully")
        
        return rows_affected
    
    def deactivate_student(self, student_db_id: int) -> int:
        """
        Deactivate a student (set is_active = 0).
        
        Args:
            student_db_id: Database primary key
        
        Returns:
            Number of rows affected (1 if deactivated, 0 if student not found)
        
        Raises:
            InvalidStudentDataError: If student_db_id is invalid
            StudentNotFoundError: If student does not exist
        """
        # Validate student_db_id
        if not isinstance(student_db_id, int) or student_db_id <= 0:
            raise InvalidStudentDataError("student_db_id must be a positive integer")
        
        # Verify student exists
        existing_student = self.repository.get_by_id(student_db_id)
        if not existing_student:
            raise StudentNotFoundError(f"Student with DB ID {student_db_id} not found")
        
        rows_affected = self.repository.deactivate_student(student_db_id)
        
        if rows_affected > 0:
            logger.info(f"Student DB ID {student_db_id} deactivated successfully")
        
        return rows_affected

    def delete_student(self, student_db_id: int) -> int:
        """
        Delete a student from the database.
        
        Args:
            student_db_id: Database primary key
            
        Returns:
            Number of rows affected (1 if deleted, 0 if not found)
            
        Raises:
            InvalidStudentDataError: If student_db_id is invalid
            StudentNotFoundError: If student does not exist
        """
        if not isinstance(student_db_id, int) or student_db_id <= 0:
            raise InvalidStudentDataError("student_db_id must be a positive integer")
            
        existing_student = self.repository.get_by_id(student_db_id)
        if not existing_student:
            raise StudentNotFoundError(f"Student with DB ID {student_db_id} not found")
            
        rows_affected = self.repository.delete_student(student_db_id)
        
        if rows_affected > 0:
            logger.info(f"Student DB ID {student_db_id} deleted successfully")
            
        return rows_affected


if __name__ == "__main__":
    # Simple manual test
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.database.schema import initialize_database
    import tempfile
    
    print("Testing StudentService...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        test_db = Path(f.name)
    
    try:
        # Initialize schema
        initialize_database(test_db)
        
        # Create service
        service = StudentService(test_db)
        
        # Test create with validation
        try:
            student_db_id = service.create_student(
                student_id="  2021001  ",  # With whitespace
                name="  John Doe  ",  # With whitespace
                email="john@example.com",
                department="Computer Science"
            )
            print(f"[OK] Created student with DB ID: {student_db_id}")
        except Exception as e:
            print(f"[ERROR] Create failed: {e}")
        
        # Test duplicate rejection
        try:
            service.create_student(
                student_id="2021001",
                name="Jane Doe"
            )
            print("[ERROR] Duplicate was not rejected")
        except DuplicateStudentError:
            print("[OK] Duplicate student_id correctly rejected")
        
        # Test invalid email
        try:
            service.create_student(
                student_id="2021002",
                name="Jane Doe",
                email="invalid-email"
            )
            print("[ERROR] Invalid email was not rejected")
        except InvalidStudentDataError:
            print("[OK] Invalid email correctly rejected")
        
        # Test get operations
        student = service.get_student_by_student_id("2021001")
        if student:
            print(f"[OK] Retrieved student: {student['name']}")
        
        # Test exists
        exists = service.student_exists("2021001")
        print(f"[OK] Student exists: {exists}")
        
        # Test update
        service.update_student(student_db_id, name="Jane Doe")
        print("[OK] Student updated")
        
        # Test deactivate
        service.deactivate_student(student_db_id)
        print("[OK] Student deactivated")
        
        # Test activate
        service.activate_student(student_db_id)
        print("[OK] Student activated")
        
        print("\n[OK] StudentService tests passed")
        
    finally:
        # Cleanup
        if test_db.exists():
            test_db.unlink()
