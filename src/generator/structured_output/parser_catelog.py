from .base_parser import BaseParser
from .json_enforcer import JSONEnforcer

class ParserCatelog:
    @classmethod
    def get(cls, parser: str | None)-> BaseParser:
        if parser == "json": return JSONEnforcer
        return None
    
# if you want a new parser just make a new file define your paser there and import it here and it