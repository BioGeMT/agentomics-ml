class IterationRunFailed(Exception):
    def __init__(self, message, context_messages, exception_trace):
        super().__init__(message)
        self.message = message
        self.context_messages = context_messages
        self.exception_trace = exception_trace

class FeedbackAgentFailed(IterationRunFailed):
    def __init__(self, message, context_messages, exception_trace):
        super().__init__(message, context_messages, exception_trace)