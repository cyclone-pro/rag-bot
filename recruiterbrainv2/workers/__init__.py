"""Background workers for async tasks."""
from .resume_processor import process_resume_upload

__all__ = ["process_resume_upload"]