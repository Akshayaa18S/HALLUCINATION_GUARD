"""Shared exception types used across the backend."""


class AppError(Exception):
    """Base class for all application-raised (as opposed to library) errors."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class NotFoundError(AppError):
    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, status_code=404)


class JobNotFoundError(NotFoundError):
    def __init__(self, job_id: str):
        super().__init__(f"Job '{job_id}' not found")


class UnauthorizedError(AppError):
    def __init__(self, message: str = "Not authenticated"):
        super().__init__(message, status_code=401)


class ForbiddenError(AppError):
    def __init__(self, message: str = "Not allowed to access this resource"):
        super().__init__(message, status_code=403)


class ConflictError(AppError):
    def __init__(self, message: str = "Resource already exists"):
        super().__init__(message, status_code=409)
