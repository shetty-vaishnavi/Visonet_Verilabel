"""
VeriLabel AI - Domain Exception Hierarchy
Provides structured error reporting across all modules.
"""

class VeriLabelException(Exception):
    """Base exception for all VeriLabel AI domain errors."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details
        }


class DocumentProcessingError(VeriLabelException):
    """Raised when document upload, format check, or image preprocessing fails."""
    pass


class OCRExtractionError(VeriLabelException):
    """Raised when OCR or structured text extraction fails."""
    pass


class SymbolDetectionError(VeriLabelException):
    """Raised during computer vision symbol detection or template loading."""
    pass


class BarcodeValidationError(VeriLabelException):
    """Raised during barcode decoding or UDI parsing/validation."""
    pass


class ComplianceRuleError(VeriLabelException):
    """Raised when compliance rules are invalid, corrupted, or cannot be evaluated."""
    pass


class DatabaseError(VeriLabelException):
    """Raised on database connection, persistence, or query execution failure."""
    pass


class ReviewWorkflowError(VeriLabelException):
    """Raised when human expert review actions fail or violate status constraints."""
    pass
