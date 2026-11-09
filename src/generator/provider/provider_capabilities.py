class ProviderCapabilities:
    def __init__(
        self,
        supports_decoding: bool,
        returns_usage: bool,
    ):
        self.supports_decoding = supports_decoding
        self.returns_usage = returns_usage
