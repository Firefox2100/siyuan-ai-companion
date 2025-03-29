class SiYuanAiCompanionError(Exception):
    def __init__(self,
                 message: str = None,
                 status_code: int = 500,
                 ):
        super().__init__(message)

        self.message = message
        self.status_code = status_code


class SiYuanApiError(SiYuanAiCompanionError):
    pass
