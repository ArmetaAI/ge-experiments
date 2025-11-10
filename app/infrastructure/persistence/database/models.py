"""
Database configuration with dual-mode connection support.

This module provides:
- SQLAlchemy engine creation (Cloud SQL Connector for production, direct URL for local dev)
- Database models (Project, ProjectFile)
- Session management
- FastAPI dependency for database access
"""

from datetime import datetime as dt
import datetime
from typing import Generator
from urllib.parse import quote_plus

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    DateTime,
    ForeignKey,
    JSON,
    Engine,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from google.cloud.sql.connector import Connector

from app.shared.config.settings import settings


# SQLAlchemy Base
Base = declarative_base()


# ============================================================================
# Database Models
# ============================================================================

class Project(Base):
    """
    Project model representing a completeness check project.
    
    Each project corresponds to one upload session and contains:
    - Unique identifier
    - Current processing status
    - Final results in JSON format
    - Processing logs for real-time monitoring
    - Metadata fields for project categorization
    - Timestamps for tracking
    """
    __tablename__ = "projects"

    id = Column(String(36), primary_key=True, index=True)  # UUID format
    status = Column(
        String(50),
        default="uploaded",
        nullable=False,
        comment="Status: uploaded, processing, completed, failed"
    )
    results_json = Column(
        JSON,
        nullable=True,
        comment="Final comparison results and report"
    )
    logs = Column(
        JSON,
        default=list,
        nullable=False,
        comment="Processing logs with timestamps for real-time monitoring"
    )
    
    # Metadata fields
    project_name = Column(
        String(512),
        nullable=True,
        comment="Project name (название проекта)"
    )
    category = Column(
        String(128),
        nullable=True,
        comment="Project category (категория проекта)"
    )
    complexity_level = Column(
        String(128),
        nullable=True,
        comment="Complexity level (уровень сложности)"
    )
    responsibility_class = Column(
        String(128),
        nullable=True,
        comment="Responsibility class (класс ответственности)"
    )
    source = Column(
        String(50),
        nullable=True,
        comment="Data source: G-BIM or manual (источник: G-BIM/ручной ввод)"
    )
    region = Column(
        String(128),
        nullable=True,
        comment="Project region (регион)"
    )
    
    created_at = Column(
        DateTime,
        default=dt.now(datetime.timezone.utc),
        nullable=False,
        index=True
    )
    updated_at = Column(
        DateTime,
        default=dt.now(datetime.timezone.utc),
        onupdate=dt.now(datetime.timezone.utc),
        nullable=False
    )
    
    # Relationship to project packages
    packages = relationship(
        "ProjectPackage",
        back_populates="project",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Project(id={self.id}, status={self.status})>"


class ProjectFile(Base):
    """
    ProjectFile model representing individual files uploaded for a project.
    
    Each file record contains:
    - Reference to parent project
    - Original filename and GCS storage path
    - Validation status for completeness check
    """
    __tablename__ = "project_files"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    package_id = Column(
        Integer,
        ForeignKey("project_packages.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    original_filename = Column(
        String(512),
        nullable=False,
        comment="Original name of the uploaded file"
    )
    gcs_path = Column(
        String(1024),
        nullable=False,
        comment="Object path within GCS bucket (e.g., 'projects/{id}/{type}/{filename}')"
    )
    validation_status = Column(
        String(50),
        default="pending",
        nullable=False,
        comment="Status: pending, matched, not_found, error"
    )
    matched_doc_number = Column(
        String(256),
        nullable=True,
        comment="Document number from ОПЗ table that matched this file"
    )
    matched_doc_name = Column(
        String(512),
        nullable=True,
        comment="Document name from ОПЗ table that matched this file"
    )
    match_score = Column(
        Integer,
        nullable=True,
        comment="Fuzzy matching score (0-100)"
    )
    compliance_results = Column(
        JSON,
        nullable=True,
        comment="Compliance check results: check_format, page_number, empty_lists, insufficient_files, classify"
    )
    created_at = Column(
        DateTime,
        default=dt.now(datetime.timezone.utc),
        nullable=False
    )
    
    # Relationship to package
    package = relationship("ProjectPackage", back_populates="files")

    def __repr__(self):
        return f"<ProjectFile(id={self.id}, filename={self.original_filename}, status={self.validation_status})>"


class ProjectPackage(Base):
    """
    ProjectPackage model representing a document package within a project.

    Supports separate tracking for:
    - ПСД (Проектно-сметная Документация)
    - ИРД (Исходно-разрешительная Документация)
    """
    __tablename__ = "project_packages"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    project_id = Column(
        String(36),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    package_type = Column(
        String(10),
        nullable=False,
        comment="Package type: PSD (Проектно-сметная Документация) or IRD (Исходно-разрешительная Документация)"
    )
    status = Column(
        String(50),
        default="uploaded",
        nullable=False,
        comment="Package status: uploaded, processing, completed, failed"
    )
    results_json = Column(
        JSON,
        nullable=True,
        comment="Package-specific validation results"
    )
    logs = Column(
        JSON,
        default=list,
        nullable=False,
        comment="Processing events log for real-time monitoring"
    )
    created_at = Column(
        DateTime,
        default=dt.now(datetime.timezone.utc),
        nullable=False
    )
    updated_at = Column(
        DateTime,
        default=dt.now(datetime.timezone.utc),
        onupdate=dt.now(datetime.timezone.utc),
        nullable=False
    )

    # Relationships
    project = relationship("Project", back_populates="packages")
    files = relationship(
        "ProjectFile",
        back_populates="package",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<ProjectPackage(id={self.id}, type={self.package_type}, status={self.status})>"


class DocumentTag(Base):
    """
    DocumentTag model for storing document classification tags with vector embeddings.

    Used for RAG-based document classification using semantic similarity search.
    Each tag contains:
    - Tag name and description
    - Related keywords for context
    - Vector embedding (768-dimensional) for similarity search
    """
    __tablename__ = "document_tags"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    tag_name = Column(
        String(256),
        nullable=False,
        index=True,
        comment="Tag name for document classification"
    )
    description = Column(
        Text,
        nullable=True,
        comment="Description of the tag"
    )
    keywords = Column(
        JSON,
        nullable=True,
        comment="Related keywords for better matching"
    )
    embedding = Column(
        Text,
        nullable=True,
        comment="Vector embedding (768-dimensional) stored as pgvector type"
    )
    created_at = Column(
        DateTime,
        default=dt.now(datetime.timezone.utc),
        nullable=False
    )
    updated_at = Column(
        DateTime,
        default=dt.now(datetime.timezone.utc),
        onupdate=dt.now(datetime.timezone.utc),
        nullable=False
    )

    def __repr__(self):
        return f"<DocumentTag(id={self.id}, tag_name={self.tag_name})>"


# ============================================================================
# Database Connection Setup
# ============================================================================

# Global connector instance (lazy-loaded)
connector = None


def getconn():
    """
    Create a connection to Cloud SQL using the Python Connector.
    
    This function is used by SQLAlchemy's create_engine creator parameter.
    The connector handles:
    - Automatic IAM authentication (when running in GCP)
    - SSL/TLS encryption
    - Connection pooling
    - Automatic IP whitelisting
    
    Returns:
        Connection: Database connection object
    """
    global connector
    if connector is None:
        connector = Connector()
    
    # Try to use IAM authentication if password is problematic
    # IAM auth is more secure and doesn't have password encoding issues
    try:
        # Get password from settings (supports both DB_PASS and DB_PASSWORD_GOSEXPERT)
        password = settings.database_password
        
        if not password:
            print("[Database] Warning: No password found, attempting IAM auth...")
            conn = connector.connect(
                settings.INSTANCE_CONNECTION_NAME,
                "pg8000",
                user=settings.DB_USER,
                db=settings.DB_NAME,
                enable_iam_auth=True,
            )
            return conn
        
        # Clean password - remove whitespace and check for control characters
        password = password.strip()
        
        # Check for control characters
        if any(ord(c) < 32 and c not in '\t\n\r' for c in password):
            print("[Database] Warning: Password contains control characters, attempting IAM auth...")
            # Try IAM authentication
            conn = connector.connect(
                settings.INSTANCE_CONNECTION_NAME,
                "pg8000",
                user=settings.DB_USER,
                db=settings.DB_NAME,
                enable_iam_auth=True,
            )
            return conn
        
        # Normal password authentication
        conn = connector.connect(
            settings.INSTANCE_CONNECTION_NAME,
            "pg8000",
            user=settings.DB_USER,
            password=password,
            db=settings.DB_NAME,
        )
        return conn
    except Exception as e:
        print(f"[Database] Connection error: {str(e)}")
        print(f"[Database] User: {settings.DB_USER}, DB: {settings.DB_NAME}")
        print(f"[Database] Instance: {settings.INSTANCE_CONNECTION_NAME}")
        print(f"[Database] Has password: {bool(settings.database_password)}")
        raise


def create_db_engine() -> Engine:
    """
    Create SQLAlchemy engine with dual-mode support.
    
    Two modes:
    1. LOCAL DEV: DATABASE_URL via Cloud SQL Proxy (postgresql+psycopg2://...)
    2. PRODUCTION: Cloud SQL Python Connector with automatic IAM auth
    
    Returns:
        Engine: SQLAlchemy engine instance
    """
    if settings.DATABASE_URL:
        # Local development: Direct connection via Cloud SQL Proxy
        print(f"[Database] Local dev mode - using DATABASE_URL")
        engine = create_engine(
            settings.DATABASE_URL,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False,
        )
    else:
        # Production: Cloud SQL Connector with IAM authentication
        print(f"[Database] Production mode - Cloud SQL Connector: {settings.INSTANCE_CONNECTION_NAME}")
        engine = create_engine(
            "postgresql+pg8000://",
            creator=getconn,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False,
        )
    
    return engine


# Create global engine instance
engine = create_db_engine()

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


# ============================================================================
# FastAPI Dependencies
# ============================================================================

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database sessions.
    
    Creates a new database session for each request and ensures proper cleanup.
    
    Usage:
        @app.get("/projects")
        def read_projects(db: Session = Depends(get_db)):
            return db.query(Project).all()
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
