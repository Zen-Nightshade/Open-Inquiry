from .usage_stats import UsageStats

class ProviderResponse():
    def __init__(self, text: str, usage: UsageStats = None, raw: object = None):
        self.text = text
        self.usage = usage
        self.raw = raw