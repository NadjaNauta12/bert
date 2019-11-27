class Error(Exception):
    """Base class for other exceptions"""
    pass


class EvaluationFileAlreadyExists(Error):
    """Raised when trying to write a file which already exists"""
    pass


class InputLengthExceeded(Error):
    """Raised when trying to write a file which already exists"""
    pass


