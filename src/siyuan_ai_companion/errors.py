"""
SiYuan Ai Companion Errors
"""

class SiYuanAiCompanionError(Exception):
    """
    Base class for all exceptions raised by the SiYuan Ai Companion.
    """
    def __init__(self,
                 message: str = None,
                 status_code: int = 500,
                 ):
        super().__init__(message)

        self.message = message
        self.status_code = status_code


class SiYuanApiError(SiYuanAiCompanionError):
    """
    Error calling SiYuan API.
    """
