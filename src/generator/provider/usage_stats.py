class UsageStats():
    def __init__(
            self, 
            prompt_tokens: int = None,
            completion_tokens: int = None,
            total_tokens: int = None    
        ):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
