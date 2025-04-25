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


class SiYuanFileListError(SiYuanApiError):
    """
    Error listing files in from SiYuan server.
    """


class SiYuanBlockNotFoundError(SiYuanApiError):
    """
    Error when a block is not found.
    """

class RagDriverError(SiYuanAiCompanionError):
    """
    Base class for all exceptions raised by the RAG driver.
    """
